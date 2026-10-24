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


BETAS_KEY_ATOL = 2e-5          # the cache key's tolerance: write_json (hc.py_json) stores every float rounded to 5 decimals


def betas_key(fit: dict) -> dict:
    """{role: betas rounded to 5 decimals} - the key capsules.json / sites.json are cached under (5 decimals: what write_json stores)"""
    return {r: [round(float(x), 5) for x in np.asarray(fit[r]["betas"]).ravel()] for r in ROLES}


def betas_key_match(stored, key: dict, atol: float = BETAS_KEY_ATOL) -> bool:
    """the cached betas equal these (per role, same length, |difference| <= atol). 2026-10-23 (critic b7): the key was compared at 6
    decimals against a file that stores 5, so the cache never hit and every call refitted and rewrote capsules.json / sites.json"""
    if not isinstance(stored, dict):
        return False
    for r in ROLES:
        a, b = stored.get(r), key.get(r)
        if a is None or b is None or len(a) != len(b):
            return False
        if not np.allclose(np.asarray(a, float), np.asarray(b, float), rtol=0.0, atol=atol):
            return False
    return True


def capsule_bodies(fit: dict | None = None, force: bool = False, write: bool = True) -> dict:
    """{role: capsule body} (cached in v4/contacts/capsules.json, keyed by the betas; force: refit; write=False: never write the cache)"""
    p = OUT / "capsules.json"
    fit = fit or load_fit("input")
    key = betas_key(fit)
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        if betas_key_match(d.get("betas"), key):
            return {r: _caps_from_json(d["bodies"][r]) for r in ROLES}
    bodies = {r: fit_capsules(fit[r]["betas"]) for r in ROLES}
    if write:
        write_json(p, {"what": "capsule bodies fitted to each dancer's shaped SMPL-X rest mesh (zouk_contacts.fit_capsules): point specs, "
                               "radii, and per capsule the error of its surface against the vertices it stands for", "betas": key,
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
    return wrist_angles_R(rotvec_to_mat(pose[:, WRIST[side]]), side)


def wrist_angles_R(R: np.ndarray, side: str) -> dict:
    """wrist_angles of the wrist's local rotation matrices R (n,3,3)"""
    a, n, t = _rest_axes(side)
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
    return elbow_angles_R(rotvec_to_mat(pose[:, ELBOW[side]]), side)


def elbow_angles_R(Re: np.ndarray, side: str) -> dict:
    """elbow_angles of the elbow's local rotation matrices Re (n,3,3)"""
    a, _, _ = _rest_axes(side)
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


def site_bodies(fit: dict | None = None, force: bool = False, write: bool = True) -> dict:
    """{role: {site: ...}} (cached in v4/contacts/sites.json, keyed by the betas like capsule_bodies)"""
    p = OUT / "sites.json"
    fit = fit or load_fit("input")
    key = betas_key(fit)
    if p.exists() and not force:
        d = json.loads(p.read_text(encoding="utf-8"))
        if betas_key_match(d.get("betas"), key):
            return d["sites"]
    sites = {r: fit_sites(fit[r]["betas"]) for r in ROLES}
    if write:
        write_json(p, {"what": "contact sites (zouk_contacts.fit_sites): point + outward normal in the frame of the joint that carries it (rest "
                               "frame x = the dancer's left, y = up, z = front), found by a ray cast on the shaped rest mesh; radius = the patch",
                       "betas": key, "sites": sites})
    return sites


def cache_check(fit: dict | None = None) -> dict:
    """the cached capsules.json / sites.json against a FRESH in-memory fit (nothing written): the largest difference of every capsule radius
    and local point, every site point / normal / radius (m) -> {"capsules": {...}, "sites": {...}, "ok": bool (<= 2e-5 m everywhere)}"""
    fit = fit or load_fit("input")
    out = {"betas_key_match": {}}
    dc = json.loads((OUT / "capsules.json").read_text(encoding="utf-8"))
    ds = json.loads((OUT / "sites.json").read_text(encoding="utf-8"))
    key = betas_key(fit)
    out["betas_key_match"] = {"capsules": betas_key_match(dc.get("betas"), key), "sites": betas_key_match(ds.get("betas"), key)}
    caps = {}
    for r in ROLES:
        fresh = fit_capsules(fit[r]["betas"])
        cached = _caps_from_json(dc["bodies"][r])
        same_specs = fresh["names"] == cached["names"] and fresh["parts"] == cached["parts"]
        dr = float(np.abs(np.asarray(fresh["r"]) - cached["r"]).max())
        dp = 0.0
        kinds_ok = True
        for pa, pb in zip(fresh["A"] + fresh["B"], cached["A"] + cached["B"]):
            if pa[0] != pb[0] or pa[1] != pb[1]:
                kinds_ok = False
                continue
            if pa[0] == "loc":
                dp = max(dp, float(np.abs(np.asarray(pa[2], float) - np.asarray(pb[2], float)).max()))
            elif pa[0] == "lerp":
                kinds_ok &= pa[2] == pb[2]
                dp = max(dp, abs(float(pa[3]) - float(pb[3])))
        caps[r] = {"n": len(fresh["names"]), "same_names": bool(same_specs), "same_point_kinds": bool(kinds_ok), "max_radius_diff_m": dr,
                   "max_local_point_diff_m": dp}
    sts = {}
    for r in ROLES:
        fresh = fit_sites(fit[r]["betas"])
        cached = ds["sites"][r]
        dmax = {"p": 0.0, "n": 0.0, "radius_m": 0.0, "rest_point": 0.0}
        for nm, v in fresh.items():
            if nm not in cached:
                dmax["missing"] = dmax.get("missing", 0) + 1
                continue
            for k in ("p", "n", "rest_point", "radius_m"):
                dmax[k] = max(dmax[k], float(np.abs(np.asarray(v[k], float) - np.asarray(cached[nm][k], float)).max()))
        sts[r] = {"n": len(fresh), "n_cached": len(cached), "same_names": sorted(fresh) == sorted(cached), "max_diff": dmax,
                  "joints_equal": all(int(fresh[nm]["joint"]) == int(cached[nm]["joint"]) for nm in fresh if nm in cached)}
    out["capsules"], out["sites"] = caps, sts
    tol = BETAS_KEY_ATOL
    out["ok"] = bool(all(out["betas_key_match"].values())
                     and all(c["same_names"] and c["same_point_kinds"] and c["max_radius_diff_m"] <= tol and c["max_local_point_diff_m"] <= tol
                             for c in caps.values())
                     and all(s["same_names"] and s["joints_equal"] and max(s["max_diff"][k] for k in ("p", "n", "radius_m", "rest_point")) <= tol
                             for s in sts.values()))
    return out


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
    # v43 (2026-10-23; see the section "v43": his palm on her upper-left back, every other pose = the input, collisions only of his right
    # hand / forearm against her torso, shoulders and upper arms)
    v43: bool = False                    # the v43 targets / masks / guards
    fit_out: str = "v4"                  # the FITS name merge / curation / the gate write
    hold_overrides: tuple = ()           # [[a_s, b_s, hold], ...]: the hold labels replaced on those frames (the review's mislabels)
    cb_intervals: tuple = ()             # [[a_s, b_s], ...]: the counterbalance leans - the target is the back / top of her left shoulder
    collar_free: bool = False            # his right collar (joint 14) moves too, bounded ...
    collar_max_deg: float = 10.0         # ... softly to this
    w_collar: float = 2.0e3              # per rad^2 beyond collar_max_deg ...
    w_collar_l2: float = 20.0            # ... and per rad^2 of any collar change
    her_mask: str = ""                   # "torso_arms": her mesh = torso + shoulders + upper arms in every penetration / surface / exact term
    upper_arm_allow_m: float = 0.015     # his forearm / upper arm may sink this far into her upper arms' mesh, his hand a third of it (her left
                                         #   arm rests on his right arm in the closed hold)
    ik_start: bool = False               # the analytic arm IK onto the target track as the start (arm_ik)
    ik_hint_ulb: tuple = (0.24, -0.10, 0.06)   # the IK's elbow hint in her torso frame (x her left, y up, z her front, m from spine 3):
    ik_hint_back: tuple = (0.25, -0.08, -0.02)  # ... at her left flank (closed hold / wrapped behind her back)
    v43_slack_m: float = 0.03            # stage 2: the palm within this of its target
    v43_rate_mps: float = 0.45           # the target's change against the input palm, at most this speed
    v43_sigma_frames: float = 4.0        # the target track smoothed (her frame)
    off_m: float = 0.07                  # a problem frame: the palm further than this from its target set (back holds: back_off_m) ...
    back_off_m: float = 0.10
    front_m: float = 0.03                # ... in front of her spine-3 plane by more than this ...
    inside_m: float = 0.01               # ... inside her (torso / shoulders / upper arms) deeper than this ...
    fold_m_deg: float = 100.0            # ... or the wrist bent past this
    keep_min_frames: int = 15            # input frames on target inside a window are kept (target = the input palm) in runs of at least this
    region_tol_m: float = 0.10           # a back-view phone's target within this of the target set is used as it is (else the nearest site)
    thru_slide_m: float = 0.40
    back_fv: float = 0.1                 # a phone sees her back: her front faces away from it (cos below this)
    reach_tol_m: float = 0.015           # the IK reach test: a target the arm misses by more is unreachable
    guard_supported: bool = True         # the 2D guards skip the frames whose target a back-view phone gives (her front -> her back)
    v43_smooth_scale: float = 5.0        # the actor's smoothness (second differences of its deltas) x this
    w_speed_add: float = 4.0e3           # v43: per (m/s)^2 x s of the change against the input beyond add_soft_mps where faster than the input
    add_soft_mps: float = 0.42
    w_wrist_rot: float = 2.0e3           # per rad^2 of the wrist's local rotation per frame beyond the input's + 4 deg
    ik_smooth_frames: float = 2.5        # the IK start low-passed over this many frames


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


RAY_KEEP, RAY_INSIDE, RAY_FLOAT, RAY_OBS, RAY_THRU = 0, 1, 2, 3, 4


def ray_hit_targets(V: np.ndarray, faces: np.ndarray, C: np.ndarray, thru: np.ndarray, pc0: np.ndarray, mode: np.ndarray, lift_m: float,
                    max_slide_m: float = 0.25, float_pull_m: float = 0.06, device: str = "cpu", obs_free: bool = False,
                    thru_slide_m: float = 0.40) -> dict:
    """per frame: the ray from the phone centre C (W,3) through `thru` (W,3: the palm, or the palm moved to MAMMA's observed hand) and its
    FIRST hit on her torso (V (W,nv,3) posed, faces: her torso faces); mode per frame:
      RAY_INSIDE (the palm inside her): the hit before the palm, within max_slide_m of it -> target = hit + lift_m x normal (a slide along the
                 ray: the hand keeps its pixels in that phone and lies on the side of her the phone sees);
      RAY_FLOAT  (the palm a few cm off her back): the hit at most float_pull_m BEYOND the palm -> target on her surface;
      RAY_OBS    (the phone shows the hand elsewhere): the hit within max_slide_m of the palm -> on her surface (obs_free: else `thru`
                 itself, the 2D fix at the palm's depth, when within max_slide_m; off by default - a back hold's hand lies on her);
      RAY_THRU   (v43, critic b2: the palm OUTSIDE her, in front of her, while a phone behind her sees the hand on her back): the ray hits her
                 BEFORE the palm (t < L), within thru_slide_m of it -> target = that first hit (her back, the side the phone sees) + lift -
                 the hand keeps its pixels in that phone and moves from her front to her back;
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
    m_th = (mode == RAY_THRU) & hit & (t < L) & (dist <= thru_slide_m)
    surf = m_in | m_fl | m_ob | m_th
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
    R, t, K = z["R"][ks], z["t"][ks], z["K"][ks].copy()
    bad = np.abs(np.linalg.det(K)) < 1e-9                       # a phone without the frame (its K is empty): never used
    K[bad] = np.eye(3)
    Xc = np.einsum("kij,kj->ki", R, pc) + t
    u = np.einsum("kij,kj->ki", K, Xc)
    u = u[:, :2] / np.where(np.abs(u[:, 2:]) < 1e-9, 1.0, u[:, 2:]) + du
    used = used & ~bad
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


def masked_depth(surf, fit_lead: dict, P_w: np.ndarray, ks: np.ndarray, VOt, device: str, bs: int = 8) -> np.ndarray:
    """(W,) the deepest point of his right hand (surface + palm patch) inside her (surf: a masked GpuSurface; her posed vertices VOt) (m, >= 0)"""
    import torch
    ids = hc.rest_hand_ids(fit_lead["betas"])["right"]
    pts = np.concatenate([ids.hand, ids.palm])
    V = hc.Body(fit_lead, "cpu").forward_np(np.asarray(P_w, float), np.asarray(fit_lead["transl"], float)[ks], verts=True, ids=pts)["verts"]
    with torch.no_grad():
        q = hc.query_batched(surf, torch.as_tensor(V, device=device), VOt, bs=bs, wn_within=0.03, inside_test=True)
    return np.maximum(0.0, -q["sd"].min(1).values.cpu().numpy().astype(float))


def refine_window(fit_in: dict, poses: dict, win: dict, H: dict, obs, judge, bias: dict, surf, bodies: dict, sites: dict, cfg: ZoukCfg,
                  hcfg: HoldCfg, device: str = "cuda", speed_ref: dict | None = None, surf_mask=None) -> dict:
    """one window of the leader's right hand (see the section's comment). poses = {role: (T,55,3)} current poses (the installed motion with
    the windows adopted so far); fit_in = the INPUT fit (the speed guards compare with it). surf_mask (v43): her masked surface (torso +
    shoulders + upper arms) for the penetration / surface / exact terms."""
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
    if "ramp_l" in win:
        # v43: ramps on each side (they never reach a frame where his right hand holds hers); a retry's longer ramp (run_refine) up to that
        rl = int(min(max(int(win["ramp_l"]), ramp), int(win.get("ramp_l_max", 10 ** 6))))
        rr = int(min(max(int(win["ramp_r"]), ramp), int(win.get("ramp_r_max", 10 ** 6))))
    else:
        rl = rr = ramp
    a0, a1 = max(0, c0 - rl), min(T - 1, c1 + rr)
    ks = np.arange(a0, a1 + 1)
    full = envelope(rl, c1 - c0 + 1, rr)
    w_np = full[(a0 - (c0 - rl)):(a0 - (c0 - rl)) + len(ks)]
    w = torch.tensor(w_np, dtype=torch.float32, device=device)
    wsum = w.sum()
    f32 = dict(dtype=torch.float32, device=device)
    # her mesh (fixed) for the planes + exact checks (+ the ray targets of the final pass)
    bF = hc.Body(fit_in["follow"], device)
    VO = bF.forward_np(poses["follow"][ks], np.asarray(fit_in["follow"]["transl"], float)[ks], verts=True, batch=256)["verts"]
    VOt = torch.as_tensor(VO, device=device)
    v43 = bool(getattr(cfg, "v43", False))
    surf_x = surf_mask if (v43 and surf_mask is not None) else surf   # the surface of the penetration / surface / exact terms
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
    v43t = None
    if v43:
        # v43: the target track (keep the input / the back-view phone's "through" target / the class's sites, reach-tested by the arm IK)
        names_w = np.asarray(H["names"])[ks]
        cbi = list(cfg.cb_intervals) or counterbalance_intervals()
        cls_w = v43_classes(names_w, ks, cbi)
        kf_w = posed(fit_in["follow"], ks, poses["follow"][ks])
        kl_w = posed(fit_in["lead"], ks, poses["lead"][ks])
        depth0 = masked_depth(surf_x, fit_in["lead"], poses["lead"][ks], ks, VOt, device)
        facing_w = couple_geometry(kl_w["J"], kf_w["J"])["facing"]
        ct_ = hc.coco_targets(fit_in["lead"], "lead", "right", ks, judge, obs, list(obs.cams), bias)
        Xe_, ve_, nve_ = ct_["elbow"]
        elbow_obs = np.where((ve_ & (nve_ >= 2))[:, None], Xe_, np.nan)
        v43t = v43_window_targets(fit_in["lead"], poses["lead"][ks], kf_w, ks, names_w, cls_w, depth0, sites["follow"], cfg, hcfg, obs, VO,
                                  bF.faces, device, facing_w, surf=surf_x, ramp_w=w_np, elbow_obs=elbow_obs)
        ray = {"cam": v43t["cam"], "cam_f": v43t["cam_f"], **{k: v43t[k] for k in ("target", "normal", "pushed", "on_surface")}}
    wi = window_inputs({r: {**fit_in[r], **_pose_fields(poses[r])} for r in ROLES}, H, ks, bodies, sites, obs, cfg, hcfg, ray)
    names = wi["names"]
    hold_core = names[(ks >= c0) & (ks <= c1)]
    main_hold = max(set(hold_core.tolist()), key=hold_core.tolist().count) if len(hold_core) else "free"
    tmpl = TEMPLATE_OF_ZHOLD.get(main_hold) if (main_hold in TORSO_HOLD_SITES and not cfg.wrist_only and not cfg.follow_phones) else None
    if v43:
        cc_ = cls_w[(ks >= c0) & (ks <= c1)]
        if len(cc_) and (cc_ == "cb").mean() > 0.4:
            tmpl = "relaxed"                          # the counterbalance: his hand over the top / back of her shoulder, the fingers curved
    if v43 and cfg.collar_free:
        actor = CollarActor(fit_in["lead"], "lead", "right", ks, poses["lead"], device, tmpl)
    else:
        actor = hc.Actor(fit_in["lead"], "lead", "right", ks, poses["lead"], device, tmpl)
    if win.get("unfold_only") and cfg.freeze_arm_unfold:
        actor.freeze_arm()                              # round r9: only the wrist turns (the shoulder / elbow deltas stay 0)
    _ce = np.flatnonzero(w_np > 0.99)
    actor.core = (int(_ce[0]), int(_ce[-1])) if len(_ce) else None
    ik_rep = None
    if v43 and cfg.ik_start and v43t is not None and actor.core is not None:
        # the analytic IK onto the target track as the start (the elbow at her left flank, a hinge elbow, the roll split)
        ik_rep = set_actor_start(actor, v43t["P_init"], actor.core)
    usable = wi["usable"]
    if cfg.follow_phones:
        # MAMMA's hand in the other phones is often inconsistent with the ray phone (the input fit is already their compromise): the 2D term
        # keeps the ray phone's view only; the independent COCO elbow (every phone) still holds the arm
        usable = usable & (np.asarray(obs.cams, dtype=object)[None, :] == np.asarray(ray["cam_f"], dtype=object)[:, None])
    twod = VisTwoD(torch, obs, actor, device, list(obs.cams), usable)
    if v43:
        # her mesh masked to torso + shoulders + upper arms; his forearm may sink upper_arm_allow_m into her upper arms (his hand a third)
        planes = MaskedPlanes(torch, surf_x, vallow=her_upper_arm_allow(cfg.upper_arm_allow_m))
    else:
        planes = hc.Planes(torch, surf)
    # capsules: his right hand / forearm / upper arm (from the actor's FK); hers (fixed)
    capL = bodies["lead"]
    i_hand = caps_index(capL, "hand_R")
    i_farm = caps_index(capL, "forearm_R")
    i_uarm = caps_index(capL, "upper_arm_R")
    capF = bodies["follow"]
    tgt_parts = ["hand_L", "hand_R", "forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R", "shoulder_L", "shoulder_R", "neck_head"]
    arm_parts = ("forearm_L", "forearm_R", "upper_arm_L", "upper_arm_R")
    if v43:
        # v43: the capsules only of his right hand / forearm against her upper arms (nothing of the hand holds, her forearms, hands, head)
        tgt_parts = ["upper_arm_L", "upper_arm_R"]
        arm_parts = ("upper_arm_L", "upper_arm_R")
    hold_hand = (names == "hand_hold")
    j_tgt = np.concatenate([caps_index(capF, p) for p in tgt_parts])
    j_arms = np.concatenate([caps_index(capF, p) for p in arm_parts])
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
    hand_cap_allow = (cfg.upper_arm_allow_m / 3.0) if v43 else 0.0       # v43: his hand may touch her upper arm by this much
    capL_sub = _sub_caps(capL, np.concatenate([i_hand, i_farm]))
    # targets (v43: the track already stands lift_m off her surface)
    site_t = torch.tensor(np.nan_to_num(wi["site_pt"] + (0.0 if v43 else hcfg.palm_lift_m) * np.nan_to_num(wi["site_n"])), **f32)
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
    if cfg.follow_phones or v43:
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
    site_w_t = torch.tensor(np.where(seen & (not cfg.follow_phones) & (not v43), cfg.site_seen_scale, 1.0) * wi["has_site"], **f32)
    reg_slack = cfg.ray_slack_m if cfg.follow_phones else (cfg.v43_slack_m if v43 else cfg.region_slack_m)
    # speed caps (hand_contact's): the input's p99 of the hand or its own speed there, + 0.1
    J_in = np.asarray(fit_in["lead"]["joints"], float)[ks]
    ref = (speed_ref or {}).get("lead_right", {"wrist": 2.0, "palm": 2.2})
    tr_in = hc.hand_tracks(J_in, "right")
    caps_sp = [torch.tensor(np.maximum(ref[k], hc.speeds(tr_in[k])) + 0.1, **f32) for k in ("wrist", "palm")]
    wr_rot = torch.tensor((3.0, 3.0, 3.0) if main_hold in TORSO_HOLD_SITES else (8.0, 8.0, 4.0), **f32)
    if v43:
        wr_rot = wr_rot * 0.2                          # v43: the arm is meant to move (the IK start); the target and the limits lead
        _oin = posed(fit_in["lead"], ks, hc.full_pose(fit_in["lead"])[ks])
        Xin_t = {"wrist": torch.tensor(_oin["J"][:, HW["right"]], **f32), "palm": torch.tensor(hc.palm_centre(_oin["J"], "right"), **f32),
                 "elbow": torch.tensor(_oin["J"][:, HE["right"]], **f32), "tips": torch.tensor(_oin["tips"][:, 5:10], **f32)}
        wr_in_lim = torch.tensor(np.radians(_local_rot_speed(poses_in["lead"][ks], WRIST["right"]) + 4.0), **f32)
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
        ov = torch.relu(cfg.cap_margin_m - c - allow - hand_cap_allow)
        l_hand = (ov ** 2).sum((1, 2)) * (0.0 if v43 else 1.0)          # v43: his hand against her upper arms = the masked mesh
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
            if v43 and planes.allow is not None:
                # v43: her upper arms give way a little (her left arm rests on his right arm): the forearm / upper arm by the allowance,
                # the hand by a third of it
                sd = sd + planes.allow * planes.near * torch.cat([torch.full((nh,), 1 / 3.0, device=device),
                                                                  torch.ones(na + nu, device=device),
                                                                  torch.full((sd.shape[1] - nh - na - nu,), 1 / 3.0, device=device)])[None]
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
            # v43: the ADDED speed (against the input) is the ramps' business only - in the core the hand rests on her whatever the input
            # hand does (its own speed is capped below, the track's speed against her by the target's rate limit)
            add_w = (1.0 - w[1:]).clamp_min(0.0) + 0.1 if v43 else 1.0
            for (X, X0), cap in zip(((J[:, HW["right"]], actor.wrist0), (hc._palm_centre_t(J, "right"), actor.palm_c0)), caps_sp):
                dX = X - X0
                dv = (dX[1:] - dX[:-1]) * HZ
                lsp = lsp + cfg.w_speed * (add_w * torch.relu(torch.sqrt((dv ** 2).sum(-1) + 1e-10) - rc.v_free_mps) ** 2).sum() / HZ
                v = (X[1:] - X[:-1]) * HZ
                lsp = lsp + cfg.w_speed * (torch.relu(torch.sqrt((v ** 2).sum(-1) + 1e-10) - cap) ** 2).sum() / HZ
            if cfg.w_speed_tips > 0:
                # the elbow and the five fingertips too (the gate's whip test): their added motion against the start beyond v_free_mps
                for X, X0 in ((o["tips5"], tips0), (J[:, HE["right"]][:, None], actor.elbow0[:, None])):
                    dv = ((X - X0)[1:] - (X - X0)[:-1]) * HZ
                    aw = add_w[:, None] if v43 else 1.0
                    lsp = lsp + cfg.w_speed_tips * (aw * torch.relu(torch.sqrt((dv ** 2).sum(-1) + 1e-10) - rc.v_free_mps) ** 2).sum() / HZ
                    if v43:
                        # ... and their own speed in the core: no faster than the input there or 0.8 m/s (no whips, no pops)
                        v_ = ((X[1:] - X[:-1]) * HZ).norm(dim=-1)
                        v0_ = ((X0[1:] - X0[:-1]) * HZ).norm(dim=-1)
                        lsp = lsp + cfg.w_speed_tips * (torch.relu(v_ - torch.clamp(v0_, min=0.8) - 0.1) ** 2).sum() / HZ
            if v43:
                # the review's speed rule as a loss: the motion's change against the INPUT may not move faster than ~0.45 m/s where the hand
                # (wrist, palm, elbow, fingertips) moves faster than the input (a calm hand resting on her while the input's flails is fine)
                pts_ = {"wrist": J[:, HW["right"]], "palm": hc._palm_centre_t(J, "right"), "elbow": J[:, HE["right"]], "tips": o["tips5"]}
                for k_, X in pts_.items():
                    Xi = Xin_t[k_]
                    a_ = ((X - Xi)[1:] - (X - Xi)[:-1]) * HZ
                    v_ = ((X[1:] - X[:-1]) * HZ).norm(dim=-1)
                    vi_ = ((Xi[1:] - Xi[:-1]) * HZ).norm(dim=-1)
                    gate_ = torch.sigmoid((v_ - vi_ + 0.1) / 0.05)
                    lsp = lsp + cfg.w_speed_add * (gate_ * torch.relu(a_.norm(dim=-1) - cfg.add_soft_mps) ** 2).sum() / HZ
            loss = loss + lsp
            terms["speed"] = lsp.detach()
            acc = actor.delta[2:] - 2 * actor.delta[1:-1] + actor.delta[:-2]
            lsm = rc.w_smooth * (cfg.v43_smooth_scale if v43 else 1.0) * (acc ** 2).sum((1, 2)).mean()
            if v43:
                # v43: the wrist's local rotation turns no faster than the input's + 4 deg / frame (the gate's flick test is + 8)
                Rw_ = actor.chain_R(w)[:, 2]
                cw_ = ((Rw_[:-1].transpose(1, 2) @ Rw_[1:]).diagonal(dim1=1, dim2=2).sum(-1) - 1) / 2
                ang_ = torch.acos(cw_.clamp(-1 + 1e-6, 1 - 1e-6))
                lsm = lsm + cfg.w_wrist_rot * (torch.relu(ang_ - wr_in_lim) ** 2).sum()
            for X in (J[:, HW["right"]], pc):
                ax_ = X[2:] - 2 * X[1:-1] + X[:-2]
                lsm = lsm + rc.w_smooth_pos * (ax_ ** 2).sum(-1).mean()
            loss = loss + lsm
            terms["smooth"] = lsm.detach()
        lrot = ((actor.delta ** 2).sum(-1) * wr_rot).sum(-1).mean()
        loss = loss + lrot
        terms["rot"] = lrot.detach()
        if hasattr(actor, "collar_k"):
            # v43: the collar (protraction / elevation) bounded: softly to collar_max_deg, a little for any change
            cang = torch.sqrt((actor.collar_delta ** 2).sum(-1) + 1e-12) * w
            lcol = cfg.w_collar * (torch.relu(cang - np.radians(cfg.collar_max_deg)) ** 2).mean() + cfg.w_collar_l2 * (cang ** 2).mean()
            loss = loss + lcol
            terms["collar"] = lcol.detach()
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
        o0 = actor.forward(w) if ik_rep is None else actor.forward(torch.zeros_like(w))     # v43's IK start is not the 'before'
        if unf is not None:                             # the 'before' measures (exact, 2D) of the motion as it was, not the unfolded start
            a0_ = hc.Actor(fit_in["lead"], "lead", "right", ks, poses_in["lead"], device, None)
            o0 = a0_.forward(torch.zeros_like(w))
            del a0_
    res = {"core": [int(c0), int(c1)], "frames": [int(a0), int(a1)], "ramp": ramp, "hold": main_hold, "template": tmpl, "unfold": unf,
           "holds": {h: int((hold_core == h).sum()) for h in np.unique(hold_core)}, "elbow_coco_share": float(ve.mean()),
           "phones_usable_2d_mean": float(usable.sum(1).mean()), "ramp_l": int(rl), "ramp_r": int(rr)}
    # v43: the 2D guards skip the frames whose target a back-view phone gives (a move from her front to her back the phone supports)
    core_g = core_np.copy()
    if v43t is not None:
        res["v43"] = v43t["stats"]
        res["thru_frames"] = [int(k) for k in ks[v43t["supported"] & core_np]]
        res["ik_start"] = ik_rep
        if cfg.guard_supported:
            core_g = core_np & ~np.asarray(v43t["supported"], bool)
    g2d = int(core_g.sum()) >= 10                     # enough frames left for the 2D guards
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
            q = hc.query_batched(surf_x, P, VOt, bs=8, wn_within=0.03, inside_test=cfg.exact_check)
            return torch.relu(-q["sd"].min(1).values).cpu().numpy().astype(float) * 100
        pe0, pe1 = exact_pen(o0), exact_pen(o1)
        if v43:
            # v43: his forearm / upper arm against her masked mesh (torso, shoulders, upper arms - her upper arms give way upper_arm_allow_m):
            # the exact version of the forearm-vs-her-arm check (the capsules over-reach by ~1 cm and near her shoulder)
            vall = torch.as_tensor(her_upper_arm_allow(cfg.upper_arm_allow_m).astype(np.float32), device=device)

            def exact_arm(o):
                P = torch.cat([o["arm"], o["upper"]], 1).detach().float()
                q = hc.query_batched(surf_x, P, VOt, bs=8, wn_within=0.03, inside_test=cfg.exact_check)
                d = torch.relu(-q["sd"] - vall[q["vert"]])
                return d.max(1).values.cpu().numpy().astype(float) * 100
            pa0, pa1 = exact_arm(o0), exact_arm(o1)
            res["exact_arm_cm"] = {"before_p90": round(pct(pa0, core_np), 2), "after_p90": round(pct(pa1, core_np), 2),
                                   "before_gt2": int((pa0[core_np] > 2).sum()), "after_gt2": int((pa1[core_np] > 2).sum())}
    res["exact_pen_cm"] = {"before_p90": round(pct(pe0, core_np), 2), "after_p90": round(pct(pe1, core_np), 2),
                           "before_max": round(float(pe0[core_np].max()), 2) if core_np.any() else 0.0,
                           "after_max": round(float(pe1[core_np].max()), 2) if core_np.any() else 0.0}
    # 2D, visibility-aware (MAMMA hand points in the usable phones) and the independent COCO elbow
    res["ma2d_vis_px"] = [twod.core_median(o0["lm"], core_g, actor.n_hand_lm) if (twod.items and g2d) else None,
                          twod.core_median(o1["lm"], core_g, actor.n_hand_lm) if (twod.items and g2d) else None]

    def agreeing(o) -> tuple[int, dict]:
        """phones x core frames whose median hand-point error is below agree_px (the usable phones); the per-phone medians"""
        n, per = 0, {}
        with torch.no_grad():
            for cam, e, v in twod.errors(o["lm"]):
                e, v = e[:, :actor.n_hand_lm], v[:, :actor.n_hand_lm]
                ev = torch.where(v, e, torch.full_like(e, float("nan")))
                med = torch.nanmedian(ev, dim=1).values.cpu().numpy()
                ok = (v.sum(1).cpu().numpy() >= 5) & core_g
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
            vis_j[nm_] = joint_visible(obs, ks[core_g], kl1["J"][core_g, jj_],
                                       bodies, {k_: v_[core_g] for k_, v_ in kl1.items()}, {k_: v_[core_g] for k_, v_ in wi["kf"].items()})
        res["vis_phone_frames"] = {k_: int(v_.sum()) for k_, v_ in vis_j.items()}

    def coco_joint_px(J, which):
        jj_ = HE["right"] if which == "elbow" else HW["right"]
        b = np.asarray(bias.get(f"lead_right_{which}", [0, 0, 0]), float)
        X = J[:, jj_] + np.einsum("kij,j->ki", hc._forearm_frame(J, "right"), b)
        if which in vis_j:
            Jo = M1["J"] if J is M0["J"] else M0["J"]
            Xo = Jo[:, jj_] + np.einsum("kij,j->ki", hc._forearm_frame(Jo, "right"), b)
            Xl = M0["J"][:, WRIST["left"] if which == "wrist" else ELBOW["left"]]
            e = coco_errors_vis(judge, obs, X[core_g], ks[core_g], which, vis_j[which], Xl[core_g], Xo[core_g])
        else:
            e = hc._coco_errors(judge, obs, X[core_g], ks[core_g], "lead", "right", which, obs.cams)
        return (float(np.median(e)), len(e)) if (len(e) and g2d) else (None, 0)

    def coco_elbow_px(J):
        return coco_joint_px(J, "elbow")[0]
    res["coco_elbow_px"] = [coco_elbow_px(M0["J"]), coco_elbow_px(M1["J"])]
    # guards
    why = []
    ncore = int(core_np.sum())
    if v43 and "exact_arm_cm" in res:
        ea_ = res["exact_arm_cm"]
        if ea_["after_p90"] > max(ea_["before_p90"], 1.0) + 1.0 or ea_["after_gt2"] > ea_["before_gt2"] + 3 + 0.06 * ncore:
            why.append(f"his forearm into her (exact, masked): p90 {ea_['before_p90']} -> {ea_['after_p90']} cm, frames > 2 cm "
                       f"{ea_['before_gt2']} -> {ea_['after_gt2']}")
    for k in ("hand_limbs_cm", "arm_arms_cm", "self_cm"):
        if (k == "self_cm" and v43 and cfg.w_self <= 0) or (k in ("hand_limbs_cm", "arm_arms_cm") and v43):
            continue                                  # v43: no capsule of his own forearm (his wrist limits stay); his hand / forearm vs her:
                                                      #   the masked mesh (exact_pen_cm, exact_arm_cm; the capsule terms stay in the loss)
        b_, a_ = res["before"][k], res["after"][k]
        sl = cfg.guard_cap_slack_cm if k != "arm_arms_cm" else (2.0 if v43 else 1.0)
        if a_["p95"] > max(b_["p95"], cfg.pen_cm) + sl or a_["frames_gt_2"] > b_["frames_gt_2"] + 3 + (0.06 if v43 else 0.02) * ncore:
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
    # v43: MAMMA's hand points are what the input was fitted to (its hand on her front at a reachable depth): reported, not a veto there;
    # the independent COCO wrist / elbow (the phones that see the joint, the back-phone-supported frames left out) stay the 2D vetoes
    notes = []
    if g2d and na0 > 0 and na1 < (1 - cfg.guard_agree_drop) * na0:
        (notes if v43 else why).append(f"fewer phones agree with the hand (MAMMA points < {cfg.agree_px:.0f} px): {na0} -> {na1} phone-frames")
    b2, a2 = res["ma2d_vis_px"]
    if b2 is not None and a2 is not None and np.isfinite(b2) and np.isfinite(a2) and a2 > b2 + cfg.guard_ma2d_slack_px:
        (notes if v43 else why).append(f"MAMMA hand points (usable phones) {b2:.1f} -> {a2:.1f} px")
    res["guard_notes"] = notes
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
        if cw0 is not None and cw1 is not None and min(n0, n1) >= GATE["coco_min_n"] and                 cw1 > cw0 + max(cfg.guard_coco_slack_px if v43 else GATE["coco_worse_px"], cfg.guard_coco_rel * cw0):
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
    res["new_pose"] = (ks, pose1, list(getattr(actor, "collar", [])) + list(actor.chain) + list(actor.fj))
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
    if cfg.hold_overrides:                           # v43: the review's mislabels
        H = holds_with_overrides(H, cfg.hold_overrides)
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
        surf_m = hc.GpuSurface(faces, len(hc.vertex_joint()), device, vmask=her_vmask(cfg.her_mask)) if cfg.her_mask else None
        for n_, i in enumerate(idx):
            w = dict(wins[i])
            te = time.time()
            attempts = []
            res = None
            for attempt in range(3):
                try:
                    res = refine_window(fit_in, poses, w, H, obs, judge, bias, surf, bodies, sites, cfg, hcfg, device, speed_ref, surf_m)
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
        rep["written"] = write_fit_v4(base, {"lead": P, "follow": hc.full_pose(base["follow"])}, fit_dir(cfg.fit_out),
                                      note=fit_note(cfg.fit_out, cfg.base), base_dir=FITS.get(cfg.base) if cfg.fit_out != "v4" else None)
    write_json(rd / "refine.json", rep)
    log(f"merged {summ}")
    return rep


def fit_dir(name: str) -> Path:
    """the directory a pass writes its fit to: FITS[name] ("v4" = v4/smplx_contact, "v43" = v4/smplx_contact_v43)"""
    return Path(FITS.get(name, FIT_OUT))


def fit_note(name: str, base: str) -> str:
    """the provenance appended to the fit's "source" field"""
    if name == "v43":
        return (f" | v43 (dancecap.zouk_contacts, 2026-10-23): his right arm re-posed onto her upper-left back in the torso-hold windows, every "
                f"other pose = {base} (smplx_rf2_noslip)")
    return " | v4 zouk contacts + collisions (dancecap.zouk_contacts, 2026-10-22) of v3/smplx_contact"


def write_fit_v4(base: dict, poses: dict, out_dir: Path, note: str | None = None, base_dir: Path | None = None) -> dict:
    """the contract npz: the installed fit's keys with the refined poses and the FK joints recomputed on the changed frames (note: the
    provenance appended to "source", default the v4 one; base_dir: the base fit's folder, for the record)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    note = note if note is not None else fit_note("v4", "v3")
    info = {}
    for r in ROLES:
        d = dict(base[r])
        P = poses[r]
        T = len(P)
        changed = np.flatnonzero(np.abs(P - hc.full_pose(base[r])).max((1, 2)) > 0)
        ref_dir = Path(base_dir) if base_dir is not None else hc.FIT_OUT          # the installed v3 (v4), the input (v43)
        changed_v3 = np.flatnonzero(np.abs(P - hc.full_pose(load_fit(ref_dir)[r])).max((1, 2)) > 0)
        d["global_orient"] = P[:, 0].copy()
        d["body_pose"] = P[:, 1:22].reshape(T, 63).copy()
        d["left_hand_pose"] = P[:, 25:40].reshape(T, 45).copy()
        d["right_hand_pose"] = P[:, 40:55].reshape(T, 45).copy()
        J = np.asarray(base[r]["joints"], float).copy()
        if len(changed):
            J[changed] = posed(base[r], changed, P[changed])["J"]
        d["joints"] = J
        src = str(base[r].get("source", ""))
        if note.strip(" |")[:12] not in src:
            src += note
        d["source"] = np.array(src)
        np.savez(out_dir / f"{r}.npz", **d)
        info[r] = {"frames_changed_vs_installed": int(len(changed_v3)), "frames_changed_this_pass": int(len(changed)), "file": str(out_dir / f"{r}.npz")}
    info["fk_check_mm"] = fk_check_v4(out_dir, ref=base_dir)
    what = ("v3/smplx_contact (installed) + the v4 zouk contact / collision resolution of the leader's right hand (dancecap.zouk_contacts)"
            if "v43" not in note else "v43: the input (smplx_rf2_noslip) for both dancers + his right arm on her upper-left back in the torso-hold "
                                      "windows (dancecap.zouk_contacts, section v43)")
    write_json(out_dir / "zouk_contacts.json", {"what": what, "base": str(base_dir or hc.FIT_OUT), "info": info})
    return info


def fk_check_v4(d: Path, n: int = 400, ref: Path | None = None) -> dict:
    """max |smplx FK joints - stored joints| (mm) on a sample of changed (vs ref, default v3/smplx_contact) + random frames, with the smplx
    package (the exporter's contract)"""
    from . import mamma_dyn as md
    out = {}
    for r in ROLES:
        z = dict(np.load(d / f"{r}.npz", allow_pickle=True))
        src = dict(np.load(Path(ref or hc.FIT_OUT) / f"{r}.npz", allow_pickle=True))
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


def validate_final(fits=("input", "v3", "v4"), changed_vs: str = "v3", path: Path | None = None, notes: dict | None = None,
                   json_path: Path | None = None) -> dict:
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
    write_json(json_path or (OUT / "validate_final.json"), rep)
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
    lab = {"v4": "final", "ec8e": "installed before (ec8e7101)", "inst2": "installed (d034e0b7)", "inst": "installed (5ccf839e)"}
    if "v43" in fits:
        lab = {"v4": "v4 (v4.2, 8bd1d1ad)", "v43": "v43"}
    rows = [["measure"] + [lab.get(f, f)
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


def export_motion(fit: str = "v4") -> dict:
    """v4/smplx_contact through export_unity into the SIDE capture work/<take>/export/06_LarissaKaduRecap_v4contacts/ (install=False: the
    Unity project is not touched), its four motion files copied to v4/contacts/unity_motion/ with their sha256 (install_motion.sh checks
    them), and a read-only comparison with the installed capture. fit="v43": export_motion_v43"""
    if fit == "v43":
        return export_motion_v43()
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


def curation_file(fit_out: str = "v4") -> Path:
    """the curation of a fit's passes: curation.json (v4), curation_<fit>.json (v43, ...)"""
    return CURATION if fit_out in ("v4", "", None) else OUT / f"curation_{fit_out}.json"


def load_curation(path: Path | None = None) -> dict:
    """v4/contacts/curation.json: {"veto": {"<pass>:<window>": {"core_s": [a, b], "why": ...}}, "blend_s": s} - windows put back to the
    installed motion after looking at their phone crops (each with its reason); the v3 curation (hand_contact) is not applied here: the v4
    windows start from the installed motion, which already carries it"""
    path = path or CURATION
    if path.exists():
        c = json.loads(path.read_text(encoding="utf-8"))
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


def apply_curation(tag: str = "pass2", write: bool = True, base_tag: str | None = None, fit_out: str = "v4") -> dict:
    """the adopted result (runs/<tag>/poses_merged.npz) with the vetoed windows put back to the installed motion: on each vetoed window's
    core the installed pose, blended in / out over blend_s by a raised cosine (slerp per joint of the leader's right arm and fingers, so the
    neighbouring windows keep their correction and nothing jumps) -> v4/smplx_contact/.
    base_tag (a STAGE, the final pass): the pass `tag` was solved from runs/<base_tag>/poses_merged.npz (e.g. "inst", the motion installed
    2026-10-22); only curation.json["stages"][tag] applies (its vetoes go back to that motion, its despike ranges are taken out of it), the
    earlier vetoes are already inside it"""
    from .hand_kin import SHOULDER as HS, ELBOW as HE, WRIST as HW, FINGER_JOINTS
    cur = load_curation(curation_file(fit_out))
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
    if fit_out != "v4":
        joints = [COLLAR["right"]] + joints                    # v43: the collar moves too
    ks = np.flatnonzero(m > 0)
    for j in joints:
        kj = ks[np.abs(P[ks, j] - P3[ks, j]).max(1) > 0] if fit_out != "v4" else ks     # v43: frames already equal stay bit-identical
        P[kj, j] = _slerp_rotvec(P[kj, j], P3[kj, j], m[kj])
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
        rep["written"] = write_fit_v4(base, {"lead": P, "follow": hc.full_pose(base["follow"])}, fit_dir(fit_out),
                                      note=fit_note(fit_out, base_tag or "v3"),
                                      base_dir=FITS.get(base_tag or "v3") if fit_out != "v4" else None)
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
            rl, rr = int(r.get("ramp_l", ramp)), int(r.get("ramp_r", ramp))
            out.append({"id": f"{tag}:{r.get('window', i)}", "core": [c0, c1], "span": [max(0, c0 - rl), c1 + rr], "hold": r.get("hold"),
                        "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)],
                        # a vetoed window goes back to the installed motion over its whole span (core + its ramps): a veto of the core alone
                        # kept a blend of the rejected solve in the ramps
                        "t_span": [round(max(0, c0 - rl) / HZ, 2), round((c1 + rr) / HZ, 2)],
                        # v43: the frames a back-view phone's target led (the 2D checks skip them)
                        "skip_frames": list(r.get("thru_frames") or [])})
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
            ks_ = np.setdiff1d(ks, np.asarray(w.get("skip_frames") or [], int))
            if len(w.get("skip_frames") or []):
                if len(ks_) < 10:
                    out[w["id"]] = {"n": 0, "v3_px": None, "v4_px": None, "vis": True, "skipped": "every core frame led by a back-view phone"}
                    continue
                ks = ks_
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


def window_check_vetoes(chk: dict, wins: list[dict], skip=(), exempt: dict | None = None, coco_rel: float = 0.0,
                        min_px: float | None = None) -> dict:
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
        if "coco_wrist" not in ex and c["n"] >= g["coco_min_n"] and c["v3_px"] is not None and c["v4_px"] is not None and                 c["v4_px"] - c["v3_px"] > max(min_px or g["coco_worse_px"], coco_rel * c["v3_px"]):
            why.append(f"the independent COCO wrist {c['v3_px']} -> {c['v4_px']} px (n {c['n']})")
        if "coco_elbow" not in ex and c.get("elbow_n", 0) >= g["coco_min_n"] and c.get("elbow_v4_px") is not None and c.get("elbow_v3_px") is not None and \
                c["elbow_v4_px"] - c["elbow_v3_px"] > max(min_px or g["coco_elbow_worse_px"], coco_rel * c["elbow_v3_px"]):
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


def exact_window_vetoes(name: str, ref: str, wins: list[dict], H: dict, skip=(), parts=None, tolerant: bool = False) -> tuple[dict, dict]:
    """per window span, from exact_<name>.npz vs exact_<ref>.npz: NEW penetration (deeper by > pen_new_cm on >= pen_new_frames frames) and a
    hand that now floats off her (gap > float_cm where the reference touched or was inside her, gap <= float_ref_cm, >= float_frames frames,
    torso holds) -> (vetoes, per-window numbers)"""
    with np.load(OUT / f"exact_{name}.npz") as z1, np.load(OUT / f"exact_{ref}.npz") as z0:
        p1, p0, g1, g0 = z1["hand_pen"], z0["hand_pen"], z1["hand_gap"], z0["hand_gap"]
        if parts:                                       # v43: only her torso / shoulders / upper arms count
            pn = [str(x) for x in z1["parts"]]
            ii = [pn.index(p_) for p_ in parts]
            p1, p0 = z1["hand_pen_part"][:, ii].max(1), z0["hand_pen_part"][:, ii].max(1)
    return exact_window_checks(p0, p1, g0, g1, H["names"], wins, skip, tolerant)


def exact_window_checks(p0: np.ndarray, p1: np.ndarray, g0: np.ndarray, g1: np.ndarray, names: np.ndarray, wins: list[dict],
                        skip=(), tolerant: bool = False) -> tuple[dict, dict]:
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
        n_span = b - a + 1
        if tolerant:
            # v43 (the hand re-posed from her front / inside her onto her back): a window is vetoed when it leaves MORE frames deeper than
            # 2 cm than the input (+3) while some got deeper, or floats on more than a tenth of its span
            n0, n1 = nums[w["id"]]["pen_frames_gt2cm"]
            if deeper >= g["pen_new_frames"] and n1 > n0 + 3:
                why.append(f"new penetration (exact): {deeper} frames deeper by > {g['pen_new_cm']} cm, frames > 2 cm {n0} -> {n1}")
            if floats > max(g["float_frames"], 0.1 * n_span):
                why.append(f"the hand now floats off her (exact gap > {g['float_cm']} cm) on {floats} frames")
        else:
            if deeper >= g["pen_new_frames"]:
                why.append(f"new penetration (exact): {deeper} frames deeper by > {g['pen_new_cm']} cm")
            if floats >= g["float_frames"]:
                why.append(f"the hand now floats off her (exact gap > {g['float_cm']} cm) on {floats} frames")
        if why:
            out[w["id"]] = {"core_s": w.get("t_span", w.get("t")), "why": "gate: " + "; ".join(why), "gate": True}
    return out, nums


def run_gate(tag: str = "pass3", rounds: int = 4, coco: bool = True, extra_veto: dict | None = None, tags=None, base_tag: str | None = None,
             exact: bool = False, device: str = "cuda", vis: bool = False, fit_out: str = "v4") -> dict:
    """apply the curation, flag NEW problems frame by frame, veto every adopted window whose span holds one (and, with coco, every window whose
    independent COCO wrist got more than GATE["coco_worse_px"] worse than v3), apply again; repeat until nothing is flagged -> gate.json,
    curation.json (the gate's vetoes marked "gate"), v4/smplx_contact/.
    base_tag (the final pass, a stage): the reference is that motion (runs/<base_tag>/, e.g. "inst", the motion installed before it), the vetoes
    go to curation.json["stages"][tag]; the window checks add the COCO elbow and MAMMA's hand points, and with exact the exact mesh check of
    the candidate (exact_update on the changed frames) adds new penetration / a newly floating hand -> gate_<tag>.json"""
    cur_path = curation_file(fit_out)
    cur_raw = json.loads(cur_path.read_text(encoding="utf-8")) if cur_path.exists() else {}
    cur_raw.setdefault("veto", {})
    if base_tag:
        stage = cur_raw.setdefault("stages", {}).setdefault(tag, {})
        stage["base_tag"] = base_tag
        if fit_out == "v43":
            # v43: the back motion is the input with the arm despikes of its two glitches (the frames no adopted window keeps); the stages
            # the written fit carries: this one on top of those of the frozen base (freeze_v43)
            stage.setdefault("despike", [list(d) for d in V43_DESPIKE])
            cur_raw["fit_stages"] = list((cur_raw.get("frozen_stages") or {}).get(base_tag, [])) + [tag]
            if base_tag == "input" and not (OUT / "runs" / "input" / "poses_merged.npz").exists():
                make_input_run("input")
        vetoes = stage.setdefault("veto", {})
    else:
        vetoes = cur_raw["veto"]
    for k, v in (extra_veto or {}).items():
        vetoes[k] = v
    cur_path.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    wins = adopted_windows(tags or ([tag] if base_tag else None))
    ref = base_tag or "v3"
    hist = []
    coco_rep = None
    exact_rep = None
    exact_done = not exact
    H = load_holds()
    v43 = fit_out == "v43"
    if v43:
        H = holds_with_overrides(H, V43_HOLD_OVERRIDES)
    for rnd in range(rounds):
        apply_curation(tag, base_tag=base_tag, fit_out=fit_out)
        fits = {"input": load_fit("input"), "ref": load_fit(ref), "cand": load_fit(fit_out)}
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
            if v43:
                # v43: a fold / twist / elbow flag on a few frames of a long re-pose is not a reason to give the whole window up (the
                # whips and flicks - pops - stay strict)
                tol = max(3, int(0.02 * (b - a + 1)))
                hit = {k: n for k, n in hit.items() if k in ("whip", "wrist_flick") or n > tol}
            if hit:
                new[w["id"]] = {"core_s": w["t_span"], "why": f"gate round {rnd + 1}: new problems in its span: {hit}", "gate": True}
        if coco and coco_rep is None:
            live = [w for w in wins if w["id"] not in vetoed and w["id"] not in new]
            coco_rep = coco_window_check(fits, live, ref="ref", cand="cand", mamma=bool(base_tag), vis=vis)
            keep = (cur_raw.get("stages", {}).get(tag, {}).get("keep") or {}) if base_tag else {}
            exm = {k: v.get("exempt", []) for k, v in keep.items()}
            if v43:                                         # v43: MAMMA's points (the input's own 2D) are reported, not a veto
                exm = {w["id"]: list(set(exm.get(w["id"], [])) | {"mamma"}) for w in live}
            new.update(window_check_vetoes(coco_rep, live, exempt=exm, coco_rel=(0.4 if v43 else GATE["coco_rel_vis"]) if vis else 0.0,
                                           min_px=8.0 if v43 else None))
        if not new and not exact_done:
            exact_update(fit_out, ref, device)
            live = [w for w in wins if w["id"] not in vetoed]
            keep_ = (cur_raw.get("stages", {}).get(tag, {}).get("keep") or {}) if base_tag else {}
            ev, exact_rep = exact_window_vetoes(fit_out, ref, live, H, skip=[k for k, v in keep_.items() if "exact" in v.get("exempt", [])],
                                                parts=V43_EXACT_PARTS if v43 else None, tolerant=v43)
            new.update(ev)
            exact_done = True
        hist.append({"round": rnd + 1, "flagged_frames": {k: int(v.sum()) for k, v in fl.items()}, "flagged_any": int(anyf.sum()),
                     "flagged_times_s": [round(x / HZ, 2) for x in np.flatnonzero(anyf)[:60]], "new_vetoes": sorted(new),
                     "frames_changed_vs_reference": int(gf["changed"].sum())})
        log(f"gate round {rnd + 1}: flagged {hist[-1]['flagged_frames']}, new vetoes {sorted(new)}")
        if not new:
            break
        vetoes.update(new)
        cur_path.write_text(json.dumps(cur_raw, indent=1, ensure_ascii=False), encoding="utf-8")
    rep = {"what": "zouk_contacts post-hoc gate: NEW problems of the candidate vs the reference and the input (twist, elbow hyperextension / "
                   "off-hinge, wrist fold, elbow / wrist / palm / fingertip whips, wrist flicks) and per window the independent COCO wrist "
                   "(+ COCO elbow, MAMMA hand points, exact new penetration / floating in the final pass)", "gate": GATE, "tag": tag,
           "reference": ref, "windows_adopted": len(wins), "windows_vetoed": sorted(k for k in vetoes), "rounds": hist,
           "coco_per_window": coco_rep, "exact_per_window": exact_rep, "windows_kept": [w for w in wins if w["id"] not in vetoes]}
    if exact_done and exact:
        # the curation's last round may have changed the candidate after the exact check: recompute where it differs from the reference
        apply_curation(tag, base_tag=base_tag, fit_out=fit_out)
        exact_update(fit_out, ref, device)
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


def last_pass(base: str = "ec8e", out: str = "v4", cfg: LastCfg | None = None, write: bool = True, only: list | None = None,
              v43: bool = False, v43_parts=None, v43_mask: str = "torso_arms", stage: str = "last43") -> dict:
    """the last pass (see the section): the base motion (the installed ec8e7101) -> steps 1-3 -> FITS[out] (v4/smplx_contact), exact_<out>.npz
    (recomputed where the leader changed), v4/contacts/last/report.json (every spike / window with its numbers and 2D).
    v43 (2026-10-23): the same on the v43 fit - the depth is her torso / shoulders / upper arms only (V43_EXACT_PARTS) and her surface is
    masked to them, only frames of the torso holds may change (never a frame where his right hand holds hers: the spans stop 3 frames short,
    the core gives way), spikes are arm-only (his body stays the input), the report in v4/contacts/last43/ and the windows kept recorded as
    the curation_v43.json stage "last43" (runs/last43/refine.json) -> FITS[out]"""
    from scipy.spatial.transform import Rotation as Ro  # noqa: F401
    cfg = cfg or LastCfg()
    t0 = time.time()
    fit = load_fit(base)
    T = len(fit["lead"]["t"])
    P0 = hc.full_pose(fit["lead"])
    tr0 = np.asarray(fit["lead"]["transl"], float)
    with np.load(OUT / f"exact_{base}.npz") as z:
        ex0 = {k: z[k].copy() for k in z.files}
    last_dir = (OUT / stage) if v43 else LAST_DIR
    pidx = [list(map(str, ex0["parts"])).index(p_) for p_ in (v43_parts or V43_EXACT_PARTS)]

    def penf(ex):
        """the depth a pass works on: her torso / shoulders / upper arms (v43) or any part of her"""
        return ex["hand_pen_part"][:, pidx].max(1) if v43 else ex["hand_pen"]
    allowed = np.ones(T, bool)
    hh = np.zeros(T, bool)
    if v43:
        Hv = holds_with_overrides(load_holds(), V43_HOLD_OVERRIDES)
        hh = np.asarray(Hv["names"]) == "hand_hold"
        allowed = (v43_classes(Hv["names"], np.arange(T), counterbalance_intervals()) != "")
        hd = hh.copy()
        for d_ in range(1, 4):
            hd[d_:] |= hh[:-d_]
            hd[:-d_] |= hh[d_:]
        allowed &= ~hd
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
    spikes = spike_ranges(penf(ex0), bend0, speeds, cfg)
    if v43:                                         # arm-only spikes inside the torso holds (his body and the hand holds stay the input)
        spikes = [dict(s_, kind="arm") for s_ in spikes if allowed[s_["a"]:s_["b"] + 1].all()]
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
        s["pen_cm_after"] = round(float(penf(ex1)[x].max()) * 100, 1)
        s["bend_deg_after"] = round(float(wrist_angles(P1[x], "right")["bend"].max()), 1)
        s["2d"] = _elbow_guard({"before": fit, "after": fit1}, x, obs, judge, bias, bodies, hcfg)
    rep["spikes"] = spikes
    rep["step1_frames_changed"] = int(len(ch1))
    log(f"last pass step 1: {len(spikes)} spikes, {len(ch1)} frames ({time.time() - t0:.0f} s)")
    # ---- steps 2 / 3: push out, unfold
    bend1 = wrist_angles(P1, "right")["bend"]
    # the contact to keep: where the INSTALLED motion's hand touched her (a despike may have lifted it); the back holds' floating frames
    # in a window are pulled onto her (step 4)
    wins = last_windows(penf(ex1), ex0["hand_gap"], bend1, cfg, back=np.isin(load_holds()["names"], BACK_HOLDS))
    if v43:
        # only frames of the torso holds are pushed / unfolded; the spans stop 3 frames short of his right hand's hand holds
        w2 = []
        for w in wins:
            w = dict(w)
            w["push"] = [k for k in w["push"] if allowed[k]]
            w["unfold"] = [k for k in w["unfold"] if allowed[k]]
            need = sorted(w["push"] + w["unfold"])
            if not need:
                continue
            c0, c1 = max(w["core"][0], need[0] - cfg.pad), min(w["core"][1], need[-1] + cfg.pad)
            c0, c1, rl_, rr_, _, _ = ramps_clear_of(c0, c1, cfg.ramp, hh, cfg.ramp, 3)
            if c1 < c0:
                continue
            w["core"], w["span"] = [c0, c1], [c0 - rl_, c1 + rr_]
            w["t"] = [round(c0 / HZ, 2), round(c1 / HZ, 2)]
            xs_ = set(range(w["span"][0], w["span"][1] + 1))
            for k_ in ("push", "unfold", "touch", "pull"):
                w[k_] = [k for k in w[k_] if k in xs_ and (k_ in ("touch", "pull") or c0 <= k <= c1)]
            w2.append(w)
        wins = w2
    rest_J = rest_body(fit["lead"]["betas"])["J"]
    pts = _arm_points(fit["lead"]["betas"])
    fit_in = load_fit("input")
    P_in = hc.full_pose(fit_in["lead"])
    o_in = posed(fit_in["lead"], None, P_in)
    p99_in = {k: float(np.percentile(hc.speeds(x), 99)) for k, x in
              (("E", o_in["J"][:, ELBOW["right"]]), ("W", o_in["J"][:, WRIST["right"]]), ("C", hc.palm_centre(o_in["J"], "right")))}
    p99_in["tips"] = float(min(np.percentile(hc.speeds(o_in["tips"][:, 5 + i]), 99) for i in range(5)))
    surf = hc.GpuSurface(hc.Body(fit["lead"], "cpu").faces, len(hc.vertex_joint()), cfg.device,
                         vmask=her_vmask(v43_mask) if v43 else None)
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
            deeper = int(((penf(ex2)[sl] > 0.02) & (penf(ex2)[sl] > penf(ex0)[sl] + 0.01)).sum())
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
        r["exact_core_depth_cm_max_after"] = round(float(penf(ex2)[x].max()) * 100, 2)
        r["exact_core_frames_gt5cm_after"] = int((penf(ex2)[x] > 0.05).sum())
        r["bend_deg_max_after"] = round(float(wrist_angles(P2[x], "right")["bend"].max()), 1)
    rep["frames_changed_vs_base"] = int(len(ch2))
    rep["seconds"] = round(time.time() - t0, 1)
    last_dir.mkdir(parents=True, exist_ok=True)
    if only:
        write_json(last_dir / "trial.json", rep)
        np.savez_compressed(last_dir / "trial_poses.npz", lead=P2, transl=tr1)
        return rep
    np.savez_compressed(last_dir / "poses_step1.npz", lead=P1, transl=tr1)
    np.savez_compressed(last_dir / "poses_final.npz", lead=P2, transl=tr1)
    if write:
        b2 = {r: dict(fit[r]) for r in ROLES}
        b2["lead"]["transl"] = tr1
        rep["written"] = write_fit_v4(b2, {"lead": P2, "follow": hc.full_pose(fit["follow"])}, FITS[out],
                                      **({"note": fit_note("v43", base), "base_dir": FITS.get(base)} if v43 else {}))
        np.savez_compressed(OUT / f"exact_{out}.npz", ks=np.arange(T), parts=ex0["parts"],
                            **{k: ex2[k] for k in ("hand_pen", "arm_pen", "hand_gap", "hand_pen_part")})
        if v43:
            # the windows this pass kept, as a curation stage of the v43 fit (the export's proof and the edge audit read them)
            kept = [{"window": i, "adopted": True, "core": list(w_["core"]), "ramp": 0, "ramp_l": int(w_["core"][0] - round(w_["t_span"][0] * HZ)),
                     "ramp_r": int(round(w_["t_span"][1] * HZ) - w_["core"][1]), "hold": "last", "t": w_["t"]}
                    for i, w_ in enumerate(rep["windows"]) if not w_.get("vetoed")]
            kept += [{"window": 1000 + i, "adopted": True, "core": [s_["a"], s_["b"]], "ramp": 0, "ramp_l": 0, "ramp_r": 0, "hold": "spike",
                      "t": s_["t"]} for i, s_ in enumerate(rep["spikes"]) if "skipped" not in s_]
            write_json(OUT / "runs" / stage / "refine.json", {"what": "the v43 last pass's kept windows (zouk_contacts.last_pass v43)",
                                                               "results": kept, "parts": list(v43_parts or V43_EXACT_PARTS),
                                                               "mask": v43_mask})
            cp = curation_file("v43")
            c = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
            c.setdefault("stages", {})[stage] = {"base_tag": base, "veto": {}}
            c["fit_stages"] = list((c.get("frozen_stages") or {}).get(base, c.get("fit_stages") or [])) + [stage]
            cp.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    write_json(last_dir / "report.json", rep)
    log(f"last pass: {len(ch2)} frames changed vs {base} ({time.time() - t0:.0f} s)")
    return rep


# ------------------------------------------------------------------------------------------------ v43 (2026-10-23)
#
# Owner (2026-10-23): "kadu's right hand should usually rest on larissa's upper left back. it is in awkward positions right now. also I think
# the colision detection is causing awkward hand holding. the hand holds should go back to the default detected poses. only the larissa
# shoulder to right hand of kadu connection should be collision checked. especially around 1:30 and 2:30 and 4:50"
#
# The fit v43 = the INPUT (the detected poses, smplx_rf2_noslip) for both dancers everywhere - her whole motion, both his hand holds, his left
# arm - except his right arm (shoulder, elbow, wrist; the collar a little, bounded; the right fingers) in the windows where his right hand is
# on her back / shoulder (the torso holds of holds.npz with the review's mislabels overridden), and the two arm despikes of the input's
# glitches. Inside a window only the frames where the input is off its target, on her front, inside her or folds the wrist are re-posed:
#   targets   upper_back / shoulder_top: his palm resting on her upper-left back (UPPER_LEFT_BACK + shoulder_blade_L), facing her; in the
#             counterbalance leans (counterbalance.json) the back / top of her left shoulder (shoulder_back_L: the blade is beyond his
#             straight arm there, critic b1), the collar free; wrap_back / behind_back: her back grid, the upper part preferred, never her
#             front; per frame a Viterbi over the candidates (keep the input / the back-view phone's target / the sites) with the arm IK's
#             reach test (an unreachable site costs), smoothed in her frame and rate-limited;
#   phones    the "through" ray: a phone that sees her back and MAMMA's hand on it -> its ray through the observed hand hits her back before
#             the palm (the input put the hand in front of her at a reachable depth) -> that hit + lift is the target (the hand keeps its
#             pixels there); the 2D guards skip those frames (a move from her front to her back that the back-view phone supports);
#   start     an analytic two-bone IK (arm_ik): the hand flat on her (palm normal into her, fingers toward her spine), the elbow at her left
#             flank (hint), a pure-hinge elbow, the forearm roll split between the elbow and the wrist (no candy wrapper);
#   collision her mesh masked to torso + shoulders + upper arms (GpuSurface vmask) for the penetration, surface and exact terms and the
#             guards; capsules only for his right hand / forearm against her upper arms (an allowance: her left arm rests on his); his
#             wrist / elbow limits as before (twist <= 85 deg, hinge, bend);
#   output    v4/smplx_contact_v43 (FITS["v43"]), curation_v43.json, export v4/contacts/unity_motion_v43/ (+ its install_motion.sh).
#
#     python -m dancecap.zouk_contacts plan-v43                     # local: the windows -> plan_v43.json (+ problems / reach per window)
#     python -m dancecap.zouk_contacts refine --parts 5 --tag h1 --write --cfg '<V43_CFG>'           (pod)
#     python -m dancecap.zouk_contacts gate --tag h1 --tags h1 --base-tag input --exact --vis --fit-out v43   (pod: exact on the GPU)
#     python -m dancecap.zouk_contacts validate-v43                 # the tables per hold / at the owner's times -> validate_v43.md
#     python -m dancecap.zouk_contacts unity-motion --fit v43        # -> v4/contacts/unity_motion_v43/ (never installed here)

FIT_V43 = V4 / "smplx_contact_v43"
FITS["v43"] = FIT_V43
# the v43 rounds after the first: the curated fit of the round before, frozen, is the next round's base (and its gate's reference)
FITS.update({f"v43{x}": V4 / f"smplx_contact_v43{x}" for x in "abcdefghij"})
UNITY_MOTION_V43 = OUT / "unity_motion_v43"
SIDE_CAPTURE_V43 = "06_LarissaKaduRecap_v43"
V43_HOLD_OVERRIDES = ((49.6, 53.8, "hand_hold"), (130.0, 142.0, "free"))     # review_check-hands.json problem [2]
V43_DESPIKE = ((165.87, 165.97), (347.50, 347.57))                         # the input's two right-wrist glitches (critic b8)
ULB5 = UPPER_LEFT_BACK + ("shoulder_blade_L",)
V43_ULB_HOLDS = ("upper_back", "shoulder_top")
V43_BACK_HOLDS = ("wrap_back", "behind_back")
HER_MASK_JOINTS = (PELVIS, SPINE1, SPINE2, SPINE3, COLLAR["left"], COLLAR["right"], SHOULDER["left"], SHOULDER["right"])
HER_RAY_JOINTS = (PELVIS, SPINE1, SPINE2, SPINE3, COLLAR["left"], COLLAR["right"])
V43_SITES = ULB5 + ("shoulder_top_L", "side_L") + BACK_GRID + ("shoulder_back_R",)
V43_EXACT_PARTS = ("torso", "shoulder_L", "shoulder_R", "upper_arm_L", "upper_arm_R")   # her parts the v43 collisions are checked against
OWNER_WINDOWS_S = ((80.0, 100.0), (140.0, 160.0), (272.8, 292.8))          # film 1:30 / 2:30 / 4:50 (+- 10 s) in dance seconds
CB_FILE = WORK / "counterbalance" / "counterbalance.json"
V43_CFG = {"base": "input", "plan": "plan_v43.json", "v43": True, "fit_out": "v43", "despike": [list(d) for d in V43_DESPIKE],
           "hold_overrides": [list(h) for h in V43_HOLD_OVERRIDES], "collar_free": True, "collar_max_deg": 15.0, "her_mask": "torso_arms",
           "ik_start": True,
           "w_cap": 6.0e4, "w_self": 0.0, "arm_arm_allow_m": 0.02, "w_twist": 1000.0, "w_elbow_lim": 1.0e4, "gate_guards": True,
           "vis_guards": True, "w_speed_tips": 4000.0, "ray_sigma_frames": 1.0, "ray_region_scale": 10.0, "w_2d": 3.0,
           "guard_coco_rel": 0.4, "guard_coco_slack_px": 6.0, "ramp_max": 60, "v43_rate_mps": 0.32, "w_wrist2d": 6.0, "w_bend": 600.0,
           "w_comp": 300.0}


def her_vmask(kind: str = "torso_arms") -> np.ndarray:
    """(nv,) bool: her vertices whose dominant skinning joint is in her torso + shoulders + upper arms ("torso_arms") or her torso + collars
    ("torso": the ray targets)"""
    js = {"torso_arms": HER_MASK_JOINTS, "torso": HER_RAY_JOINTS,
          "torso_arms_head": HER_MASK_JOINTS + (NECK, HEAD, 22, 23, 24)}[kind]
    return np.isin(hc.vertex_joint(), js)


def her_upper_arm_allow(allow_m: float) -> np.ndarray:
    """(nv,) per-vertex allowance (m): allow_m on her upper arms (dominant joint = a shoulder), 0 elsewhere"""
    return np.where(np.isin(hc.vertex_joint(), (SHOULDER["left"], SHOULDER["right"])), float(allow_m), 0.0)


def holds_with_overrides(H: dict, overrides=()) -> dict:
    """a copy of load_holds() with the hold of the frames in each [a_s, b_s, hold] replaced (site_id -1 there)"""
    H = dict(H)
    names = np.asarray(H["names"]).astype(object).copy()
    sid = np.asarray(H["site_id"]).copy()
    T = len(names)
    for ov in overrides or ():
        a, b, h = int(round(float(ov[0]) * HZ)), int(round(float(ov[1]) * HZ)), str(ov[2])
        a, b = max(0, a), min(T - 1, b)
        names[a:b + 1] = h
        sid[a:b + 1] = -1
    H["names"] = names.astype(str)
    H["site_id"] = sid
    return H


def counterbalance_intervals(path: Path | None = None, pad_s: float = 0.0) -> list:
    """the counterbalance leans whose connection is his right hand (counterbalance.json, kind "counterbalance", pair lead_r_hand-*)
    -> [[a_s, b_s], ...]"""
    p = path or CB_FILE
    if not Path(p).exists():
        return []
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    out = []
    for iv in d.get("intervals", []):
        pair = str((iv.get("connection") or {}).get("pair", ""))
        if iv.get("kind") == "counterbalance" and pair.startswith("lead_r_hand"):
            out.append([round(float(iv["t0"]) - pad_s, 3), round(float(iv["t1"]) + pad_s, 3)])
    return out


def v43_classes(names: np.ndarray, ks: np.ndarray, cb_intervals=()) -> np.ndarray:
    """per frame the v43 target class: "ulb" (upper_back / shoulder_top: her upper-left back), "cb" (those holds inside a counterbalance
    lean: the back / top of her left shoulder), "back" (wrap_back / behind_back: her back grid), "" (no torso hold: never re-posed)"""
    names = np.asarray(names)
    out = np.full(len(names), "", dtype=object)
    out[np.isin(names, V43_ULB_HOLDS)] = "ulb"
    out[np.isin(names, V43_BACK_HOLDS)] = "back"
    t = np.asarray(ks, float) / HZ
    cb = np.zeros(len(names), bool)
    for a, b in cb_intervals or ():
        cb |= (t >= float(a)) & (t <= float(b))
    out[(out == "ulb") & cb] = "cb"
    return out.astype(str)


def class_site_sets(cls: str) -> tuple:
    """the sites a palm counts as on target for a class (the problem test and the validation)"""
    if cls == "ulb":
        return ULB5
    if cls == "cb":
        return ULB5 + ("shoulder_top_L",)
    if cls == "back":
        return BACK_GRID + ("shoulder_back_L", "shoulder_back_R")
    return ()


def v43_site_points(kf: dict, sites_f: dict, lift_m: float, names=V43_SITES) -> dict:
    """{site: (point lifted lift_m off her surface (W,3), outward normal (W,3))} on her posed body"""
    out = {}
    for nm in names:
        if nm not in sites_f:
            continue
        p, n = site_world(sites_f[nm], kf["J"], kf["R"])
        out[nm] = (p + lift_m * n, n)
    return out


def v43_problem_frames(pc: np.ndarray, Jf: np.ndarray, cls: np.ndarray, pts: dict, depth: np.ndarray, bend: np.ndarray, cfg) -> dict:
    """per frame (the palm pc (W,3), her joints Jf (W,55,3), the class, her site points (v43_site_points), the palm's depth inside her torso /
    shoulders / upper arms (m), the wrist bend (deg)): why the frame needs the re-pose -> {"off", "front", "inside", "fold", "any",
    "dist_m" (to the nearest site of its set)}"""
    W = len(pc)
    cls = np.asarray(cls)
    dist = np.full(W, np.inf)
    for c in ("ulb", "cb", "back"):
        m = cls == c
        if not m.any():
            continue
        P = np.stack([pts[s][0][m] for s in class_site_sets(c) if s in pts], 1)
        dist[m] = np.linalg.norm(P - pc[m, None], axis=-1).min(1)
    loc = in_frame(pc, Jf)
    has = cls != ""
    lim = np.where(cls == "back", cfg.back_off_m, cfg.off_m)
    r = {"off": has & (dist > lim), "front": has & (loc[:, 2] > cfg.front_m), "inside": has & (np.asarray(depth) > cfg.inside_m),
         "fold": has & (np.asarray(bend) > cfg.fold_m_deg)}
    r["any"] = r["off"] | r["front"] | r["inside"] | r["fold"]
    r["dist_m"] = dist
    r["z_her_m"] = loc[:, 2]
    return r


def _unit(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    out = v / np.maximum(n, 1e-12)
    if fallback is not None:
        out = np.where(n > 1e-9, out, fallback)
    return out


def _rot_about(axis: np.ndarray, ang: np.ndarray) -> np.ndarray:
    """(W,3) unit axes, (W,) angles -> (W,3,3)"""
    return rotvec_to_mat(np.asarray(axis, float) * np.asarray(ang, float)[:, None])


def arm_rest(betas: np.ndarray, side: str = "right") -> dict:
    """the bones of an arm in the shaped rest pose: collar -> shoulder (r17), shoulder -> elbow (r19), elbow -> wrist (r21), the palm centre
    in the wrist's frame (c), the hand's rest finger direction (f) and palm normal (n)"""
    b = hc.Body({"betas": betas, "t": np.zeros(1)}, "cpu")
    Jr = b.forward_np(np.zeros((1, 55, 3)), np.zeros((1, 3)), verts=False)["joints"][0]
    co, sh, el, wr = COLLAR[side], SHOULDER[side], ELBOW[side], WRIST[side]
    return {"r17": Jr[sh] - Jr[co], "r19": Jr[el] - Jr[sh], "r21": Jr[wr] - Jr[el],
            "c": hc.palm_centre(Jr[None], side)[0] - Jr[wr], "f": hc.finger_dir(Jr[None], side)[0], "n": hc.palm_normal(Jr[None], side)[0]}


def arm_ik(J14: np.ndarray, G9: np.ndarray, G14: np.ndarray, rest: dict, X: np.ndarray, nt: np.ndarray, fdir: np.ndarray, ehint: np.ndarray,
           hinge_in: np.ndarray | None = None, side: str = "right", collar_max_deg: float = 10.0, reach_frac: float = 0.995,
           roll_share: float = 0.5, roll_max_deg: float = 85.0, bend_max_deg: float = 60.0, iters: int = 3,
           wrist_twist_deg: float = 75.0, follow_forearm: bool = False, side_from_input: np.ndarray | None = None,
           bend_soft: bool = False) -> dict:
    """the analytic arm onto a palm target, per frame: the palm centre at X (W,3), the palm (volar) facing -nt (into her surface), the fingers
    toward fdir (projected onto her surface), the elbow on the two-bone circle nearest the hint ehint, a pure-hinge elbow (its flexion axis
    the normal of the shoulder-elbow-wrist plane; near a straight arm the input's axis hinge_in), the forearm's roll split between the elbow
    (roll_share, <= roll_max_deg) and the wrist (its twist); beyond the straight arm's reach (reach_frac) the collar turns the shoulder toward
    the target (<= collar_max_deg). Where the wrist bends past bend_max_deg the fingers turn toward the forearm (iters).
    J14 (W,3) collar joint, G9 / G14 (W,3,3) global rotations of spine 3 / the collar, rest = arm_rest()
    -> {"R14", "R17", "R19", "R21" local rotations (W,3,3), "S", "E", "W", "residual_m" (wrist target missed by), "reach" (before the collar),
       "collar_deg", "theta_deg" (elbow flexion), "bend_deg", "twist_deg" (wrist), "roll_deg" (elbow)}"""
    X, nt, fdir, ehint = (np.asarray(v, float) for v in (X, nt, fdir, ehint))
    Wn = len(X)
    a, _, _ = _rest_axes(side)
    yh = _unit(np.cross(a, np.array([0, 0, 1.0])))
    r17, r19, r21, cl = (np.asarray(rest[k], float) for k in ("r17", "r19", "r21", "c"))
    fr, nr = _unit(np.asarray(rest["f"], float)), np.asarray(rest["n"], float)
    nr = _unit(nr - (nr @ fr) * fr)
    L1, L2 = float(np.linalg.norm(r19)), float(np.linalg.norm(r21))
    b1r, b2r = r19 / L1, r21 / L2
    Mr = np.stack([fr, nr, np.cross(fr, nr)], 1)
    y1 = _unit(yh - (yh @ b1r) * b1r)
    Fr = np.stack([b1r, y1, np.cross(b1r, y1)], 1)
    up = np.tile([0, 1.0, 0], (Wn, 1))
    v = -_unit(nt, up)
    pref = _unit(np.asarray(fdir, float))
    f_cur = pref.copy()

    def along_forearm(g):
        """the fingers along the forearm's direction g (on her surface), turned toward the preferred direction where that agrees"""
        gp = _unit(g - (g * v).sum(1, keepdims=True) * v, pref)
        pp = pref - (pref * v).sum(1, keepdims=True) * v
        return _unit(gp + 0.5 * pp * ((pp * gp).sum(1, keepdims=True) > 0))
    if follow_forearm:
        f_cur = along_forearm(X - ehint)                                 # the forearm's direction guessed from the elbow hint
    S0 = J14 + np.einsum("wij,j->wi", G14, r17)
    for it in range(iters):
        f_ = f_cur - (f_cur * v).sum(1, keepdims=True) * v
        alt = _unit(np.cross(v, np.cross(up, v)), _unit(np.cross(v, np.array([1.0, 0, 0]))))
        f_ = _unit(f_, alt)
        Mw = np.stack([f_, v, np.cross(f_, v)], 2)
        G21 = Mw @ Mr.T
        Wt = X - np.einsum("wij,j->wi", G21, cl)
        cw = np.einsum("wij,j->wi", G14, r17)
        Lmax = (L1 + L2) * reach_frac
        reach = np.linalg.norm(Wt - S0, axis=1) / (L1 + L2)
        alpha = np.zeros(Wn)
        G14n = G14.copy()
        need = np.linalg.norm(Wt - S0, axis=1) > Lmax
        if collar_max_deg > 0 and need.any():
            ax = _unit(np.cross(cw, Wt - J14), np.tile([0, 0, 1.0], (Wn, 1)))
            lo, hi = np.zeros(Wn), np.full(Wn, np.radians(collar_max_deg))
            for _ in range(16):
                mid = 0.5 * (lo + hi)
                Sm = J14 + np.einsum("wij,wj->wi", _rot_about(ax, mid), cw)
                ok = np.linalg.norm(Wt - Sm, axis=1) <= Lmax
                hi = np.where(ok, mid, hi)
                lo = np.where(ok, lo, mid)
            alpha = np.where(need, hi, 0.0)
            G14n = _rot_about(ax, alpha) @ G14
        S = J14 + np.einsum("wij,j->wi", G14n, r17)
        D = Wt - S
        d = np.linalg.norm(D, axis=1)
        u = _unit(D, up)
        dcl = np.clip(d, abs(L1 - L2) + 1e-3, (L1 + L2) * 0.9995)
        Weff = S + u * dcl[:, None]
        al = (L1 ** 2 - L2 ** 2 + dcl ** 2) / (2 * dcl)
        rr = np.sqrt(np.maximum(L1 ** 2 - al ** 2, 0.0))
        C0 = S + u * al[:, None]
        h = ehint - C0
        hp = h - (h * u).sum(1, keepdims=True) * u
        hn = np.linalg.norm(hp, axis=1)
        fb = _unit(np.cross(u, np.cross(-up, u)), _unit(np.cross(u, np.array([1.0, 0, 0]))))
        q = _unit(hp, fb)                                           # the elbow's side of the shoulder-wrist line (from the hint)
        if hinge_in is not None:
            # where the hint lies (nearly) on the shoulder-wrist line its side is undefined: the side that keeps the input's flexion axis
            hin = np.asarray(hinge_in, float)
            q_in = _unit(np.cross(u, hin), q)
            wq = np.clip(hn / 0.03, 0.0, 1.0)[:, None]
            if side_from_input is not None:
                wq = wq * (1.0 - np.clip(np.asarray(side_from_input, float), 0, 1))[:, None]   # the input's side (continuous in time)
            q = _unit(wq * q + (1 - wq) * q_in, q)
        E = C0 + q * rr[:, None]
        e1 = _unit(E - S)
        fd = _unit(Weff - E)
        # the flexion axis = the normal of the shoulder-elbow-wrist plane, q x u (= e1 x fd, without its precision loss near a straight arm):
        # the elbow bends forward in that plane (theta >= 0), continuous in time with the hint
        npv = _unit(np.cross(q, u), _unit(np.cross(e1, up)))
        npv = _unit(npv - (npv * e1).sum(1, keepdims=True) * e1, _unit(np.cross(e1, up)))
        Fw = np.stack([e1, npv, np.cross(e1, npv)], 2)
        G17 = Fw @ Fr.T
        fl = np.einsum("wji,wj->wi", G17, fd)                       # the forearm direction in the shoulder's frame
        bp = b2r - (b2r @ yh) * yh
        fp = fl - (fl @ yh)[:, None] * yh
        th = np.arctan2(np.cross(np.tile(bp, (Wn, 1)), fp) @ yh, fp @ bp)
        R19h = _rot_about(np.tile(yh, (Wn, 1)), th)
        # the rest forearm bone is not exactly perpendicular to the hinge axis: the smallest rotation that lands it on the wanted direction
        u1 = np.einsum("wij,j->wi", R19h, b2r)
        cx = np.cross(u1, fl)
        sx = np.linalg.norm(cx, axis=1)
        R19h = _rot_about(_unit(cx, np.tile(yh, (Wn, 1))), np.arctan2(sx, (u1 * fl).sum(1))) @ R19h
        M = np.transpose(G17 @ R19h, (0, 2, 1)) @ G21
        tw = np.unwrap(np.radians(wrist_angles_R(M, side)["twist"])) if Wn > 1 else np.radians(wrist_angles_R(M, side)["twist"])
        if Wn > 1:
            tw = tw - 2 * np.pi * np.round(np.median(tw) / (2 * np.pi))  # continuous in time, centred on (-pi, pi]
        # the elbow takes roll_share of the twist, more where the wrist would keep over wrist_twist_deg, at most roll_max_deg
        need = np.maximum(roll_share * np.abs(tw), np.abs(tw) - np.radians(wrist_twist_deg))
        tau = np.sign(tw) * np.minimum(need, np.radians(roll_max_deg))
        R19 = R19h @ _rot_about(np.tile(b2r, (Wn, 1)), tau)                # the forearm's roll about its own bone
        G19 = G17 @ R19
        R21 = np.transpose(G19, (0, 2, 1)) @ G21
        wa = wrist_angles_R(R21, side)
        if follow_forearm:
            if it < iters - 1:
                f_cur = along_forearm(fd)
                continue
            break
        big = wa["bend"] > bend_max_deg
        if not big.any() or it == iters - 1:
            break
        fdp = _unit(fd - (fd * v).sum(1, keepdims=True) * v, f_)
        if bend_soft:
            # fix round 1: continuous in the bend (a frame-to-frame switch of the fingers' direction pops the hand)
            wb = np.clip((wa["bend"] - bend_max_deg) / 25.0, 0.0, 1.0)[:, None]
            f_cur = _unit(wb * _unit(0.6 * fdp + 0.4 * f_) + (1 - wb) * f_, f_)
        else:
            f_cur = np.where(big[:, None], _unit(0.6 * fdp + 0.4 * f_), f_cur)
    R14 = np.transpose(G9, (0, 2, 1)) @ G14n
    R17 = np.transpose(G14n, (0, 2, 1)) @ G17
    return {"R14": R14, "R17": R17, "R19": R19, "R21": R21, "G21": G21, "S": S, "E": E, "W": Weff, "Wt": Wt,
            "residual_m": np.linalg.norm(Wt - Weff, axis=1), "reach": reach, "collar_deg": np.degrees(alpha), "theta_deg": np.degrees(th),
            "bend_deg": wa["bend"], "twist_deg": wa["twist"], "roll_deg": np.degrees(tau)}


def ik_pose(P: np.ndarray, ik: dict, collar: bool = True) -> np.ndarray:
    """the pose P (W,55,3) with his right arm's local rotations from arm_ik (the collar only where it turned)"""
    from scipy.spatial.transform import Rotation as Ro
    P = np.asarray(P, float).copy()
    for j, k in ((SHOULDER["right"], "R17"), (ELBOW["right"], "R19"), (WRIST["right"], "R21")):
        P[:, j] = Ro.from_matrix(ik[k]).as_rotvec()
    if collar:
        m = ik["collar_deg"] > 1e-6
        if m.any():
            P[m, COLLAR["right"]] = Ro.from_matrix(ik["R14"][m]).as_rotvec()
    return P


def v43_choose(cost: np.ndarray, allowed: np.ndarray, switch: float = 2.5) -> np.ndarray:
    """per frame the candidate (columns) of least cost among the allowed ones, with a cost per change of candidate (Viterbi: hysteresis)"""
    score = np.where(allowed, -np.asarray(cost, float), -1e6)
    C = score.shape[1]
    tr = np.full((C, C), float(switch))
    np.fill_diagonal(tr, 0.0)
    return viterbi(score, tr)


def smooth_in_frame(X: np.ndarray, Jf: np.ndarray, sigma: float) -> np.ndarray:
    """a point track X (W,3) smoothed in her torso frame (spine 3) over time (gaussian sigma frames)"""
    from scipy.ndimage import gaussian_filter1d
    if len(X) < 3 or sigma <= 0:
        return np.asarray(X, float).copy()
    Fr = torso_frame(Jf)
    loc = np.einsum("kji,kj->ki", Fr, X - Jf[:, SPINE3])
    loc = gaussian_filter1d(loc, sigma, axis=0, mode="nearest")
    return np.einsum("kij,kj->ki", Fr, loc) + Jf[:, SPINE3]


def v43_window_targets(fit_lead: dict, P: np.ndarray, kf: dict, ks: np.ndarray, names: np.ndarray, cls: np.ndarray, depth: np.ndarray,
                       sites_f: dict, cfg, hcfg: HoldCfg, obs=None, VO: np.ndarray | None = None, faces: np.ndarray | None = None,
                       device: str = "cpu", facing: np.ndarray | None = None, surf=None, ramp_w: np.ndarray | None = None,
                       elbow_obs: np.ndarray | None = None) -> dict:
    """the target track of one v43 window (see the section): his start pose P (W,55,3) at the frames ks, her posed joints kf, the hold names /
    classes, the start palm's depth inside her (masked, m); obs + VO (her posed vertices) + faces for the back-view phones' "through" targets
    (None: no phones) -> {"target", "normal", "pushed", "on_surface", "supported" (a back-view phone's target), "cam_f", "cam", "sel"
    (candidate name per frame), "P_init" (the IK start, blended in by the size of the move), "ik" (its numbers), "stats"}"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    W = len(ks)
    cls = np.asarray(cls)
    o = posed(fit_lead, ks, P)
    J, R = o["J"], o["R"]
    Jf = kf["J"]
    pc0 = hand_site(J, "right", "palm")[0]
    Fr = torso_frame(Jf)
    xh, yu, zh = Fr[:, :, 0], Fr[:, :, 1], Fr[:, :, 2]
    pts = v43_site_points(kf, sites_f, hcfg.palm_lift_m)
    bend = wrist_angles(P, "right")["bend"]
    prob = v43_problem_frames(pc0, Jf, cls, pts, depth, bend, cfg)
    has = cls != ""
    ok = has & ~prob["any"]
    ok = hc._keep_runs(ok, int(cfg.keep_min_frames)) if W > 1 else ok
    # ---- the back-view phones' "through" targets
    thru = np.zeros((W, 3))
    thru_n = np.zeros((W, 3))
    thru_ok = np.zeros(W, bool)
    cam_f = np.array([""] * W, dtype=object)
    cam_score = np.full(W, -np.inf)
    if obs is not None and VO is not None and faces is not None:
        tf = faces[her_vmask("torso")[faces].all(1)]
        lm_m = hc.hand_points_np(fit_lead, P if len(P) == len(fit_lead["t"]) else _pad_pose(fit_lead, P, ks), ks, "right")["lm"]
        lm_l = hc.hand_points_np(fit_lead, P if len(P) == len(fit_lead["t"]) else _pad_pose(fit_lead, P, ks), ks, "left")["lm"]
        hid = hc.landmark_parts()["right_hand"]
        for c in obs.cams:
            z = obs.z[c]
            Cc = -np.einsum("kji,kj->ki", z["R"][ks], z["t"][ks])
            toC = Cc - Jf[:, SPINE3]
            toC[:, 1] = 0
            fv = (toC * zh).sum(1) / np.maximum(np.linalg.norm(toC, axis=1), 1e-9)
            see, _ = hand_2d(obs, c, ks, hcfg)
            see = see & ~mamma_swapped(obs, c, ks, [lm_m], lm_l, hcfg) & (fv < cfg.back_fv) & has
            if see.sum() < 3:
                continue
            Xo, used = observed_thru(obs, c, ks, lm_m, pc0, see, hcfg)
            used = used & see
            if not used.any():
                continue
            D = Xo - Cc
            L = np.linalg.norm(D, axis=1)
            mode = np.where(used, RAY_THRU, RAY_KEEP)
            r = ray_hit_targets(np.asarray(VO, np.float32), tf, Cc, Xo, pc0, mode, hcfg.palm_lift_m, cfg.max_slide_m, cfg.float_pull_m,
                                device, thru_slide_m=cfg.thru_slide_m)
            on = r["on_surface"] & ((r["normal"] * zh).sum(1) < 0.1) & (in_frame(r["target"], Jf)[:, 2] < 0.0)
            vv = (z["vis"][ks, hc.PIDX["lead"]][:, hid] > hcfg.vis).sum(1)
            score = vv * (0.2 - fv)
            better = on & (score > cam_score)
            thru[better], thru_n[better] = r["target"][better], r["normal"][better]
            thru_ok |= better
            cam_f[better] = c
            cam_score = np.where(better, score, cam_score)
    # ---- the candidates
    cand = ["keep", "thru"] + [s for s in V43_SITES if s in pts]
    C = len(cand)
    Xc = np.zeros((W, C, 3))
    Nc = np.zeros((W, C, 3))
    cost = np.zeros((W, C))
    allowed = np.zeros((W, C), bool)
    Xc[:, 0], allowed[:, 0] = pc0, ok | ~has
    Xc[:, 1], Nc[:, 1] = thru, thru_n
    loc_pc = in_frame(pc0, Jf)
    d_thru = {}
    for j, s in enumerate(cand[2:], 2):
        Xc[:, j], Nc[:, j] = pts[s]
        d_thru[s] = np.linalg.norm(pts[s][0] - thru, axis=1)
    # the back-view phone's target within region_tol of the class's set is used as it is; else the class's sites nearest to it lead
    near_set = np.full(W, np.inf)
    for c_ in ("ulb", "cb", "back"):
        m = cls == c_
        if m.any():
            ss = [s for s in class_site_sets(c_) if s in pts]
            near_set[m] = np.min([d_thru[s][m] for s in ss], 0)
    thru_in = thru_ok & (near_set <= cfg.region_tol_m)
    allowed[:, 1] = thru_in & ~ok
    cost[:, 1] = 0.2
    ulb_rank = {"shoulder_blade_L": 1.0, "back_L2_hi": 1.1, "back_L1_hi": 1.2, "shoulder_back_L": 1.25, "back_C_hi": 1.6,
                "shoulder_top_L": 3.0, "side_L": 3.5}
    cb_rank = {"shoulder_back_L": 1.0, "back_L2_hi": 1.5, "shoulder_blade_L": 1.6, "back_L1_hi": 1.7, "back_C_hi": 2.0,
               "shoulder_top_L": 2.5, "side_L": 3.5}
    snap = thru_ok & ~thru_in                                           # the phone shows the hand on her back, outside the set
    for j, s in enumerate(cand[2:], 2):
        for c_, rank in (("ulb", ulb_rank), ("cb", cb_rank)):
            if s in rank:
                m = (cls == c_) & ~ok
                allowed[m, j] = True
                cost[m, j] = rank[s] + np.where(snap[m], 4.0 * d_thru[s][m], 0.0)
        if s in BACK_GRID or s == "shoulder_back_R":
            m = (cls == "back") & ~ok
            ls = in_frame(pts[s][0], Jf)
            ref = np.where(snap[:, None], in_frame(thru, Jf), loc_pc)
            lat = np.linalg.norm((ls - ref)[:, :2], axis=1)
            lvl = 0.0 if s.endswith("_hi") else (0.15 if s.endswith("_mid") else (0.35 if s.endswith("_lo") else 0.3))
            allowed[m, j] = True
            cost[m, j] = 1.0 + 3.0 * lat[m] + lvl
    # ---- the reach test: the IK onto every allowed site candidate; a target the arm misses by more than reach_tol_m costs 20
    rest = arm_rest(fit_lead["betas"])
    G9, G14, J14 = R[:, SPINE3], R[:, COLLAR["right"]], J[:, COLLAR["right"]]
    hinge_in = np.einsum("wij,j->wi", R[:, SHOULDER["right"]], _unit(np.cross(_rest_axes("right")[0], np.array([0, 0, 1.0]))))
    face = facing if facing is not None else np.full(W, -1.0)
    ftf = face < HoldCfg().face_to_face
    # face to face, his arm round her left side: the fingers toward her spine on her back, over the back of her shoulder on its top, toward
    # her back on her flank (the preferred direction projected onto her surface at the target)
    fdir_ftf = _unit(-0.6 * xh - 0.8 * zh + 0.25 * yu)
    hint = np.where((cls == "back")[:, None], np.einsum("kij,j->ki", Fr, np.asarray(cfg.ik_hint_back, float)),
                    np.einsum("kij,j->ki", Fr, np.asarray(cfg.ik_hint_ulb, float))) + Jf[:, SPINE3]
    # the elbow hint at her flank (face to face, not leaning) or the input's elbow, blended over ~0.4 s where the kind changes (a jump of the
    # hint would flip the upper arm)
    wf = (ftf & (cls != "cb")).astype(float)
    if W > 3:
        wf = np.clip(gaussian_filter1d(wf, 6.0, mode="nearest"), 0.0, 1.0)
    hint = wf[:, None] * hint + (1 - wf[:, None]) * J[:, ELBOW["right"]]
    side_in = 1.0 - wf                       # the elbow's side from the input's flexion axis where the hint is the input elbow
    if elbow_obs is not None:
        # where the independent COCO elbow is triangulated (>= 2 phones, within 35 cm of the input's), the elbow goes there: the detector
        # that sees the elbow leads its side (blended over ~0.3 s where it comes and goes)
        Eo = np.asarray(elbow_obs, float)
        okE = np.isfinite(Eo).all(1) & (np.linalg.norm(np.nan_to_num(Eo) - J[:, ELBOW["right"]], axis=1) < 0.35)
        wo = okE.astype(float)
        if W > 3:
            wo = np.clip(gaussian_filter1d(wo, 4.0, mode="nearest"), 0.0, 1.0) * okE.any()
        if okE.any():
            idx = np.flatnonzero(okE)
            Ef = np.stack([np.interp(np.arange(W), idx, Eo[idx, d]) for d in range(3)], 1)
            hint = wo[:, None] * Ef + (1 - wo[:, None]) * hint
            side_in = side_in * (1.0 - wo)
    S_in = J[:, SHOULDER["right"]]

    def fdir_for(X, n):
        fa = _unit(X - S_in)
        fa = _unit(fa - (fa * n).sum(1, keepdims=True) * n + 0.3 * yu)
        return np.where(ftf[:, None], fdir_ftf, fa)
    resid = np.zeros((W, C))
    for j in range(1, C):
        m = allowed[:, j]
        if not m.any():
            continue
        nn_ = _unit(Nc[:, j], -zh)
        ik = arm_ik(J14[m], G9[m], G14[m], rest, Xc[m, j], nn_[m], fdir_for(Xc[:, j], nn_)[m], hint[m], hinge_in[m],
                    collar_max_deg=cfg.collar_max_deg if cfg.collar_free else 0.0, follow_forearm=True, side_from_input=side_in[m])
        resid[m, j] = ik["residual_m"]
        # beyond the arm's reach: a step cost and more the further (the least unreachable site wins where none is reachable)
        over = np.maximum(0.0, ik["residual_m"] - cfg.reach_tol_m)
        cost[m, j] += np.where(over > 0, 6.0 + 200.0 * over, 0.0) + np.where(ik["bend_deg"] > 100.0, 2.0, 0.0)
    sel = v43_choose(cost, allowed)
    ar = np.arange(W)
    Xs, Ns = Xc[ar, sel], Nc[ar, sel]
    keep = sel == 0
    # ---- the track: the change against the input palm, smoothed in her frame and rate-limited over the CORE; on the ramps the core edge's
    # change faded by the ramp weight (what the actor's blend does: the ramp is a pure blend of the core edge's correction)
    wr = np.ones(W) if ramp_w is None else np.asarray(ramp_w, float)
    cm = wr > 0.99
    if not cm.any():
        cm = np.ones(W, bool)
    ci = np.flatnonzero(cm)
    # in the core the target lives in HER frame (it rests on her: its speed against her is bounded, whatever the input hand does)
    Xl = np.einsum("kji,kj->ki", Fr, Xs - Jf[:, SPINE3])
    xc = Xl[ci].copy()
    if len(ci) > 3:
        xc = gaussian_filter1d(xc, cfg.v43_sigma_frames, axis=0, mode="nearest")
    if cfg.v43_rate_mps > 0:
        xc = speed_limit_track(xc, cfg.v43_rate_mps / HZ)
    dc = xc - loc_pc[ci]
    dl = np.zeros_like(Xl)
    dl[ci] = dc
    dl[:ci[0]] = wr[:ci[0], None] * dc[0]
    dl[ci[-1] + 1:] = wr[ci[-1] + 1:, None] * dc[-1]
    tgt = pc0 + np.einsum("kij,kj->ki", Fr, dl)
    keep = keep | ~cm
    on = ~keep
    nl = np.einsum("kji,kj->ki", Fr, Ns)
    if on.any():
        idx = np.flatnonzero(on)
        for dd in range(3):
            nl[:, dd] = np.interp(ar, idx, nl[idx, dd])
        if W > 3:
            nl = gaussian_filter1d(nl, cfg.v43_sigma_frames, axis=0, mode="nearest")
    nrm = _unit(np.einsum("kij,kj->ki", Fr, nl), -zh)
    if surf is not None and VO is not None and on.any():
        # the track on her surface: inside her or closer than the lift -> onto the nearest point of her (masked) surface + lift (a move from
        # her front to her back goes round her flank, not through her); the palm's normal = her smooth surface normal there
        tgt, nrm = surface_lift(masked_surface("torso", surf.dev, faces) if faces is not None else surf, tgt, VO, hcfg.palm_lift_m, on, Jf,
                                nrm)
        if cfg.v43_rate_mps > 0 and len(ci) > 2:
            # the lift may step where the track passes a corner of her (the top of her shoulder): the core's speed limit once more
            xl2 = np.einsum("kji,kj->ki", Fr[ci], tgt[ci] - Jf[ci, SPINE3])
            xl2 = speed_limit_track(xl2, cfg.v43_rate_mps / HZ)
            tgt[ci] = np.einsum("kij,kj->ki", Fr[ci], xl2) + Jf[ci, SPINE3]
        dl = np.einsum("kji,kj->ki", Fr, tgt - pc0)
    move = np.linalg.norm(dl, axis=1)
    # ---- the IK start on the track, blended in by the size of the move (a 1-cm move does not reconfigure the arm)
    ik = arm_ik(J14, G9, G14, rest, tgt, nrm, fdir_for(tgt, nrm), hint, hinge_in, collar_max_deg=cfg.collar_max_deg if cfg.collar_free else 0.0,
                follow_forearm=True, side_from_input=side_in)
    P_ik = ik_pose(P, ik, collar=cfg.collar_free)
    wik = np.clip(move / 0.05, 0.0, 1.0)
    if W > 3:
        wik = np.clip(gaussian_filter1d(wik, 3.0, mode="nearest"), 0.0, 1.0)
    P_init = np.asarray(P, float).copy()
    for jj in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"]):
        P_init[:, jj] = _slerp_rotvec(P[:, jj], P_ik[:, jj], wik)
    if W > 5 and cfg.ik_smooth_frames > 0:
        # the IK start smoothed in time (quaternions, one hemisphere): a change of site or of the hand's normal reconfigures the arm
        P_init[ci] = smooth_rotations(P_init[ci], (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"]), cfg.ik_smooth_frames)
    supported = (np.asarray(cand, dtype=object)[sel] == "thru")
    sel_names = np.asarray(cand, dtype=object)[sel]
    cams = [c for c in cam_f[supported] if c]
    stats = {"frames": int(W), "classes": {c_: int((cls == c_).sum()) for c_ in ("ulb", "cb", "back", "")},
             "problems": {k: int(prob[k].sum()) for k in ("off", "front", "inside", "fold", "any")}, "kept": int(keep.sum()),
             "selected": {str(k): int(v) for k, v in zip(*np.unique(sel_names.astype(str), return_counts=True))},
             "thru_available": int(thru_ok.sum()), "thru_in_region": int(thru_in.sum()), "supported": int(supported.sum()),
             "unreachable_selected": int((resid[ar, sel] > cfg.reach_tol_m).sum()),
             "reach_ratio_p50_max": [round(float(np.median(ik["reach"][on])), 3) if on.any() else None,
                                     round(float(ik["reach"][on].max()), 3) if on.any() else None],
             "collar_deg_max": round(float(ik["collar_deg"].max()), 2), "ik_residual_cm_max": round(float(ik["residual_m"][on].max()) * 100, 2)
             if on.any() else 0.0, "move_cm_max": round(float(move.max()) * 100, 1)}
    return {"target": tgt, "normal": np.where(on[:, None], nrm, 0.0), "pushed": on, "on_surface": on, "supported": supported,
            "cam_f": cam_f, "cam": max(set(cams), key=cams.count) if cams else "", "sel": sel_names, "P_init": P_init, "ik": ik,
            "problems": prob, "stats": stats, "w_ik": wik}


def speed_limit_track(X: np.ndarray, step: float, iters: int = 200) -> np.ndarray:
    """a point track X (W,3) whose frame-to-frame change is brought to at most `step` (m) by local averaging where it is faster (the
    neighbours of a fast step are pulled toward each other; symmetric in time, no lag)"""
    X = np.asarray(X, float).copy()
    if len(X) < 3:
        return X
    for _ in range(iters):
        v = np.linalg.norm(np.diff(X, axis=0), axis=1)
        bad = np.flatnonzero(v > step * 1.0001)
        if not len(bad):
            break
        idx = np.unique(np.clip(np.concatenate([bad, bad + 1]), 1, len(X) - 2))
        X[idx] = 0.25 * X[idx - 1] + 0.5 * X[idx] + 0.25 * X[idx + 1]
        if bad.min() == 0:
            X[0] = X[1] + (X[0] - X[1]) * min(1.0, step / max(np.linalg.norm(X[0] - X[1]), 1e-12))
        if bad.max() == len(X) - 2:
            X[-1] = X[-2] + (X[-1] - X[-2]) * min(1.0, step / max(np.linalg.norm(X[-1] - X[-2]), 1e-12))
    return X


def surface_lift(surf, X: np.ndarray, VO: np.ndarray, lift_m: float, on: np.ndarray, Jf: np.ndarray, n0: np.ndarray,
                 rounds: int = 2, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """a target track X (W,3) on the frames `on`: points inside her (surf: her masked GpuSurface, posed vertices VO) or closer to her
    surface than lift_m go to the nearest surface point + lift_m along its normal, smoothed in her frame between rounds; the normal there
    (her interpolated vertex normals; n0 elsewhere) -> (X, normals)"""
    import torch
    X = np.asarray(X, float).copy()
    n = np.asarray(n0, float).copy()
    on = np.asarray(on, bool)
    Vt = torch.as_tensor(np.asarray(VO, np.float32), device=surf.dev)
    for r_ in range(rounds):
        with torch.no_grad():
            q = hc.query_batched(surf, torch.as_tensor(X[:, None].astype(np.float32), device=surf.dev), Vt, bs=16, wn_within=0.03)
        sd = q["sd"][:, 0].cpu().numpy().astype(float)
        qq = q["q"][:, 0].cpu().numpy().astype(float)
        nq = _unit(q["nq"][:, 0].cpu().numpy().astype(float))       # her smooth (interpolated vertex) outward normal there
        low = on & (sd < lift_m)
        X[low] = qq[low] + lift_m * nq[low]
        n = np.where(on[:, None], nq, n)
        if r_ < rounds - 1 and len(X) > 3:
            X = np.where(on[:, None], smooth_in_frame(X, Jf, sigma), X)
    if on.sum() > 3:
        # the palm's normal smoothed in her frame over the frames on her (a frame-to-frame flip of the hand would whip the wrist)
        from scipy.ndimage import gaussian_filter1d
        Fr = torso_frame(Jf)
        nl = np.einsum("kji,kj->ki", Fr, n)
        idx = np.flatnonzero(on)
        nl_on = gaussian_filter1d(nl[idx], 3.0, axis=0, mode="nearest")
        nl[idx] = nl_on
        n = np.einsum("kij,kj->ki", Fr, nl)
    return X, _unit(n)


_MASKED_SURF: dict = {}


def masked_surface(kind: str, device: str, faces: np.ndarray):
    """her GpuSurface masked to "torso_arms" (penetration) or "torso" (where the palm rests: torso + collars), built once per process"""
    key = (kind, str(device))
    if key not in _MASKED_SURF:
        _MASKED_SURF[key] = hc.GpuSurface(faces, len(hc.vertex_joint()), device, vmask=her_vmask(kind))
    return _MASKED_SURF[key]


def smooth_rotations(P: np.ndarray, joints, sigma: float) -> np.ndarray:
    """the local rotations of `joints` in P (W,55,3) low-passed over time (quaternions kept in one hemisphere, gaussian sigma frames)"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    P = np.asarray(P, float).copy()
    for j in joints:
        q = Ro.from_rotvec(P[:, j]).as_quat()
        for i in range(1, len(q)):
            if (q[i] * q[i - 1]).sum() < 0:
                q[i] *= -1
        q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        P[:, j] = Ro.from_quat(q).as_rotvec()
    return P


def _pad_pose(fit_lead: dict, P: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """a full-length (T,55,3) pose with P on the frames ks (the input elsewhere): hc.hand_points_np indexes poses by frame"""
    full = hc.full_pose(fit_lead).copy()
    full[ks] = P
    return full


class CollarActor(hc.Actor):
    """hand_contact.Actor with his collar (joint 14) free as well: one more local-frame delta on the same smooth basis, applied before the
    shoulder (parents first), bounded in the loss (ZoukCfg.collar_max_deg / w_collar); the rest as the Actor"""

    def __init__(self, fit_r: dict, role: str, side: str, ks: np.ndarray, pose: np.ndarray, device: str, template: str | None):
        super().__init__(fit_r, role, side, ks, pose, device, template)
        torch = self.torch
        from .hand_kin import rodrigues
        self.collar = [COLLAR[side]]
        self._chg = self.collar + self.chain + self.fj
        nk = self.B.shape[1]
        self.collar_k = torch.zeros((nk, 3), dtype=torch.float32, device=device, requires_grad=True)
        self._Rb_collar = rodrigues(torch, self.base[:, self.collar]).detach()[:, 0]

    @property
    def collar_delta(self):
        return self._edge(self.B @ self.collar_k)

    def collar_R(self, w):
        from .hand_kin import rodrigues
        return self._Rb_collar @ rodrigues(self.torch, w[:, None] * self.collar_delta)

    def params(self, stage: int):
        return [self.delta_k, self.collar_k] + ([self.u_k] if (self.u_k is not None and stage == 2) else [])

    def fast_forward(self, P, Rchain=None):
        """as hand_contact.Actor.fast_forward with the collar first: Rchain (W,4,3,3) = collar, shoulder, elbow, wrist"""
        from .hand_kin import rodrigues
        torch = self.torch
        kin = self.kin
        ch = self._chg
        Rl = rodrigues(torch, P[:, ch])
        if Rchain is not None:
            Rl = torch.cat([Rchain, Rl[:, Rchain.shape[1]:]], 1)
        newR, newP = {}, {}
        for n, j in enumerate(ch):
            par = int(kin.parents[j])
            Rp = newR[par] if par in newR else self._Rb[:, par]
            Pp = newP[par] if par in newP else self._Pb[:, par]
            newR[j] = Rp @ Rl[:, n]
            newP[j] = Pp + (Rp @ kin.rel[j][:, None])[..., 0]
        R = self._Rb.clone()
        Jp = self._Pb.clone()
        idx = torch.as_tensor(ch, device=self.dev)
        R[:, idx] = torch.stack([newR[j] for j in ch], 1)
        Jp[:, idx] = torch.stack([newP[j] for j in ch], 1)
        tl = Jp - (R @ kin.J[None, :, :, None])[..., 0]
        pf = self._pfb.clone().reshape(len(P), 54, 9)
        pf[:, idx - 1] = (Rl - self._eye).reshape(len(P), len(ch), 9)
        out = {"R": R, "joints": Jp + self.transl[:, None], "tl": tl, "pose_feature": pf.reshape(len(P), -1)}
        out["tips"] = kin._lbs(out, kin._tip, self.transl)
        out["verts"] = kin._lbs(out, kin.subset, self.transl)
        return out

    def forward(self, w):
        Rc = self.torch.cat([self.collar_R(w)[:, None], self.chain_R(w)], 1)
        o = self.fast_forward(self.pose(w), Rc)
        V = o["verts"]
        a = 0
        for key, n in (("hand", self.nh), ("palm", self.npm), ("arm", self.na), ("upper", self.nu)):
            o[key] = V[:, a:a + n]
            a += n
        o["lm"] = V[:, a:]
        o["tips5"] = o["tips"][:, :5] if self.side == "left" else o["tips"][:, 5:]
        return o

    def pose_np(self, w_np):
        from scipy.spatial.transform import Rotation as Ro
        torch = self.torch
        P = super().pose_np(w_np)
        with torch.no_grad():
            Rc = self.collar_R(torch.tensor(w_np, dtype=torch.float32, device=self.dev)).cpu().numpy().astype(np.float64)
        same = np.asarray(w_np) <= 0
        P[:, self.collar[0]] = np.where(same[:, None], self._base_np[:, self.collar[0]], Ro.from_matrix(Rc).as_rotvec())
        return P

    def snapshot(self):
        return super().snapshot() + (self.collar_k.detach().clone(),)

    def restore(self, snap):
        super().restore(snap[:2])
        with self.torch.no_grad():
            self.collar_k.copy_(snap[2])


def unwrap_rotvecs(r: np.ndarray) -> np.ndarray:
    """(W,3) rotation vectors (angles <= pi, as scipy returns them) made continuous in time: where the rotation passes 180 deg the next
    vector is taken as its equivalent of angle 2 pi - theta about the opposite axis when that is nearer the previous one"""
    r = np.asarray(r, float).copy()
    for k in range(1, len(r)):
        n = np.linalg.norm(r[k])
        if n < 1e-9:
            continue
        alt = r[k] - 2 * np.pi * r[k] / n
        if np.linalg.norm(alt - r[k - 1]) < np.linalg.norm(r[k] - r[k - 1]):
            r[k] = alt
    return r


def set_actor_start(actor, P_target: np.ndarray, core: tuple | None) -> dict:
    """set the actor's deltas (shoulder / elbow / wrist, + the collar of a CollarActor) so that its core frames take the local rotations of
    P_target (W,55,3) (least squares on the smooth temporal basis; the ramps keep the core edge's correction, as the Actor does) -> the fit's
    largest angle error (deg)"""
    import torch
    from scipy.spatial.transform import Rotation as Ro
    W = len(P_target)
    c0, c1 = core if core is not None else (0, W - 1)
    rows = np.arange(c0, c1 + 1)
    B = actor.B.detach().cpu().numpy().astype(np.float64)
    Rb = actor._Rb_chain.detach().cpu().numpy().astype(np.float64)              # (W,3,3,3)
    out = {}
    dl = np.zeros((W, 3, 3))
    for i, j in enumerate(actor.chain):
        Rt = Ro.from_rotvec(P_target[:, j])
        rel = Ro.from_matrix(Rb[:, i]).inv() * Rt
        dl[:, i] = unwrap_rotvecs(rel.as_rotvec())
    kn = np.linalg.lstsq(B[rows], dl[rows].reshape(len(rows), -1), rcond=None)[0]
    with torch.no_grad():
        actor.delta_k.copy_(torch.tensor(kn.reshape(-1, 3, 3), dtype=torch.float32, device=actor.dev))
    fitted = (B @ kn).reshape(W, 3, 3)
    out["chain_err_deg"] = round(float(np.degrees(np.linalg.norm(fitted[rows] - dl[rows], axis=-1)).max()), 2)
    if hasattr(actor, "collar_k"):
        Rbc = actor._Rb_collar.detach().cpu().numpy().astype(np.float64)
        rel = Ro.from_matrix(Rbc).inv() * Ro.from_rotvec(P_target[:, actor.collar[0]])
        dc = unwrap_rotvecs(rel.as_rotvec())
        kc = np.linalg.lstsq(B[rows], dc[rows], rcond=None)[0]
        with torch.no_grad():
            actor.collar_k.copy_(torch.tensor(kc, dtype=torch.float32, device=actor.dev))
        out["collar_err_deg"] = round(float(np.degrees(np.linalg.norm((B @ kc)[rows] - dc[rows], axis=-1)).max()), 2)
    return out


class MaskedPlanes(hc.Planes):
    """hand_contact.Planes (on a masked surface: GpuSurface vmask) that also keep, per query point, the allowance of its closest vertex (m):
    the v43 penetration terms let his forearm sink that far into her upper arms"""

    def __init__(self, torch, surf, reach: float = 0.12, vallow: np.ndarray | None = None):
        super().__init__(torch, surf, reach)
        self.vallow = None if vallow is None else torch.as_tensor(np.asarray(vallow, np.float32), device=surf.dev)
        self.allow = None

    def refresh(self, P, V, inside_test: bool = False):
        torch = self.torch
        with torch.no_grad():
            qr = hc.query_batched(self.surf, P, V, inside_test=inside_test)
            self.q, self.n = qr["q"].float(), qr["nq"].float()
            inside = qr["sd"] < 0
            side = ((P - self.q) * self.n).sum(-1)
            flip = (inside & (side > 0)) | (~inside & (side < 0))
            self.n = torch.where(flip[..., None], -self.n, self.n)
            self.near = (qr["dist"] < self.reach).float()
            self.allow = self.vallow[qr["vert"]] if self.vallow is not None else torch.zeros_like(self.near)


def ramps_clear_of(c0: int, c1: int, ramp: int, blocked: np.ndarray, need: int, min_core: int, first: bool = True,
                   last: bool = True, gap: int = 3) -> tuple:
    """a window's ramps that stop `gap` frames short of a blocked frame (his right hand holding hers): where that leaves less than `need`,
    the core gives way at that end (its first / last frames become the ramp, down to min_core frames; only at a run's first / last chunk)
    -> (c0, c1, ramp_l, ramp_r, ramp_l_max, ramp_r_max)"""
    T = len(blocked)
    prev = np.flatnonzero(blocked[:c0])
    nxt = np.flatnonzero(blocked[c1 + 1:])
    rl_max = (c0 - int(prev[-1]) - 1 - gap) if len(prev) else c0
    rr_max = (int(nxt[0]) - gap) if len(nxt) else T - 1 - c1
    if len(prev) and rl_max < need and first:
        sh = min(need - max(rl_max, 0), max(0, c1 - c0 + 1 - min_core))
        c0 += sh
        rl_max = max(rl_max, 0) + sh
    if len(nxt) and rr_max < need and last:
        sh = min(need - max(rr_max, 0), max(0, c1 - c0 + 1 - min_core))
        c1 -= sh
        rr_max = max(rr_max, 0) + sh
    rl, rr = int(max(0, min(ramp, rl_max))), int(max(0, min(ramp, rr_max)))
    return int(c0), int(c1), rl, rr, int(max(0, rl_max)), int(max(0, rr_max))


def plan_v43(cfg: ZoukCfg | None = None, out: str | None = None, pen: np.ndarray | None = None) -> dict:
    """the v43 windows (local, light): the torso holds (overrides applied) where the INPUT palm is off its class's target set, in front of
    her, inside her (exact_input.npz: torso / shoulders / upper arms, > inside_m) or the wrist folds (> fold_m_deg), merged (merge_s),
    padded (pad_s), cut into chunks of max_core_s; ramps sized by the move at the ends (the distance to the set there), never reaching a frame
    where his right hand holds her hand (the hand holds stay the input); per window the reach of the class's sites (straight arm = 1)
    -> v4/contacts/plan_v43.json"""
    cfg = cfg or ZoukCfg(**V43_CFG)
    fit = load_fit(cfg.base)
    T = len(fit["lead"]["t"])
    H = holds_with_overrides(load_holds(), cfg.hold_overrides)
    cbi = list(cfg.cb_intervals) or counterbalance_intervals()
    ks = np.arange(T)
    cls = v43_classes(H["names"], ks, cbi)
    sites = site_bodies(load_fit("input"))
    PL = hc.full_pose(fit["lead"])
    if cfg.despike:
        PL, _ = despike_poses(PL, cfg.despike)
    kl, kf = posed(fit["lead"], None, PL), posed(fit["follow"])
    pc = hand_site(kl["J"], "right", "palm")[0]
    if pen is None:
        with np.load(OUT / f"exact_{cfg.base}.npz") as z:
            parts = [str(x) for x in z["parts"]]
            pen = np.zeros(T)
            pen[z["ks"]] = z["hand_pen_part"][:, [parts.index(p) for p in ("torso", "shoulder_L", "shoulder_R", "upper_arm_L", "upper_arm_R")]].max(1)
    pts = v43_site_points(kf, sites["follow"], HoldCfg().palm_lift_m)
    bend = wrist_angles(PL, "right")["bend"]
    prob = v43_problem_frames(pc, kf["J"], cls, pts, pen, bend, cfg)
    anyp = prob["any"].copy()
    for a_s, b_s in cfg.despike or ():                               # the despiked glitches are re-solved too (inside a hold)
        a, b = int(round(float(a_s) * HZ)), int(round(float(b_s) * HZ))
        anyp[a:b + 1] |= cls[a:b + 1] != ""
    hh = H["names"] == "hand_hold"
    pad = int(round(cfg.pad_s * HZ))
    wins = []
    rest = arm_rest(fit["lead"]["betas"])
    L_arm = float(np.linalg.norm(rest["r19"]) + np.linalg.norm(rest["r21"]) + np.linalg.norm(rest["c"]))
    for a, b in _runs(anyp, int(round(cfg.merge_s * HZ))):
        a, b = max(0, a - pad), min(T - 1, b + pad)
        # no core frame in a hand hold or outside the torso holds
        inside_hold = cls[a:b + 1] != ""
        q = np.flatnonzero(inside_hold & ~hh[a:b + 1])
        if not len(q):
            continue
        a, b = a + int(q[0]), a + int(q[-1])
        if (b - a + 1) / HZ < cfg.min_core_s:
            continue
        n = int(np.ceil((b - a + 1) / (cfg.max_core_s * HZ)))
        edges = np.linspace(a, b + 1, n + 1).round().astype(int)
        for ci in range(n):
            c0, c1 = int(edges[ci]), int(edges[ci + 1]) - 1
            if ci > 0:
                c0 = max(a, c0 - cfg.ramp_min)
            ends = [x for x, at_end in ((c0, ci == 0), (c1, ci == n - 1)) if at_end]
            move = max([min(0.35, float(prob["dist_m"][x])) if np.isfinite(prob["dist_m"][x]) else 0.0 for x in ends] + [0.0])
            ramp = hc.ramp_for(move, cfg.v_ramp_mps, cfg.ramp_min, cfg.ramp_max)
            c0, c1, rl, rr, rl_max, rr_max = ramps_clear_of(c0, c1, ramp, hh, max(int(cfg.ramp_min), int(ramp)),
                                                            max(3, int(round(cfg.min_core_s * HZ))), ci == 0, ci == n - 1)
            sl = slice(c0, c1 + 1)
            S = kl["J"][sl, SHOULDER["right"]]
            reach = {}
            for s in ("shoulder_blade_L", "back_L2_hi", "shoulder_back_L", "side_L"):
                if s in pts:
                    rr_ = np.linalg.norm(pts[s][0][sl] - S, axis=1) / L_arm
                    reach[s] = [round(float(np.median(rr_)), 3), round(float((rr_ > 1.0).mean()), 3)]
            names = H["names"][sl]
            win = {"core": [c0, c1], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)], "chunk": [ci, n], "run": [a, b], "ramp": int(max(rl, rr)),
                   "ramp_l": rl, "ramp_r": rr, "ramp_l_max": int(max(0, rl_max)), "ramp_r_max": int(max(0, rr_max)),
                   "holds": {h: int((names == h).sum()) for h in np.unique(names)},
                   "classes": {c_: int((cls[sl] == c_).sum()) for c_ in ("ulb", "cb", "back", "")},
                   "why_frames": {k: int(prob[k][sl].sum()) for k in ("off", "front", "inside", "fold")},
                   "reach_median_share_gt1": reach, "move_m_ends": round(move, 3)}
            wins.append(win)
    rep = {"what": "v43 windows of the leader's right hand (zouk_contacts.plan_v43) in the motion " + cfg.base, "config": asdict(cfg),
           "cb_intervals": cbi, "arm_length_m": round(L_arm, 3),
           "frames_by_class": {c_: int((cls == c_).sum()) for c_ in ("ulb", "cb", "back")},
           "frames_with_problems": {k: int(prob[k].sum()) for k in ("off", "front", "inside", "fold", "any")},
           "windows": wins, "window_seconds": round(sum(w["core"][1] - w["core"][0] + 1 for w in wins) / HZ, 1)}
    write_json(OUT / (out or cfg.plan), rep)
    np.savez_compressed(OUT / "problems_v43_input.npz", cls=cls, **{k: prob[k] for k in ("off", "front", "inside", "fold", "any", "dist_m", "z_her_m")})
    log(f"plan v43: {len(wins)} windows, {rep['window_seconds']} s; problems {rep['frames_with_problems']}")
    return rep


INPUT_EXPORT = hc.V3 / "unity_motion" / "backup_installed_motion"   # the input's export, installed 2026-10-11 (lead 86e6780b, follow 98657745)
INPUT_FOLLOW_SHA = "986577451dc1bb2667e8b650351349d211664d88d556c4be61f3e0e536e1e6dc"
V43_ARM_JOINTS = (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55))


def export_motion_v43() -> dict:
    """v4/smplx_contact_v43 through export_unity into the SIDE capture work/<take>/export/06_LarissaKaduRecap_v43/ (install=False), its four
    motion files copied to v4/contacts/unity_motion_v43/ with their sha256, the proof against the input's export (prove_v43_export) and its
    own install_motion.sh (its first install backs up the motion installed then - v4.2's - so --restore returns it)"""
    import hashlib
    import shutil
    from . import export_unity as eu
    rep = eu.export(TAKE, SIDE_CAPTURE_V43, install=False, room=False, textures=False, faces=False, shape_edits=True, allow_no_audio=True,
                    smplx_dir="v4/smplx_contact_v43")
    side = WORK / "export" / SIDE_CAPTURE_V43
    UNITY_MOTION_V43.mkdir(parents=True, exist_ok=True)
    for f in hc.MOTION_FILES:
        shutil.copy2(side / f, UNITY_MOTION_V43 / f)
    out = {"side_export": str(side), "files": [str(UNITY_MOTION_V43 / f) for f in hc.MOTION_FILES],
           "fk_max_err_mm": {r: d["fk_max_err_mm"] for r, d in rep["dancers"].items()}, "warnings": rep["warnings"],
           "sha256": {f: hashlib.sha256((UNITY_MOTION_V43 / f).read_bytes()).hexdigest() for f in hc.MOTION_FILES},
           "fit": str(FIT_V43), "exported": time.strftime("%Y-%m-%d %H:%M:%S")}
    out["proof_vs_input_export"] = prove_v43_export(UNITY_MOTION_V43)
    out["installed_now"] = installed_now_list(eu.UNITY_STREAMING / hc.UNITY_CAPTURE)
    write_json(UNITY_MOTION_V43 / "unity_motion.json", out)
    _write_install_script_v43()
    return out


def installed_now_list(inst: Path) -> list:
    """the motion files installed in the capture now: [{"file", "sha256"}] - a list, never a {"<file>": "<sha256>"} object (fix round 1,
    2026-10-23: the manifest held two such objects, so install_motion.sh's grep found two hashes per file and refused every install)"""
    import hashlib
    return [{"file": f, "sha256": hashlib.sha256((Path(inst) / f).read_bytes()).hexdigest()} for f in hc.MOTION_FILES
            if (Path(inst) / f).exists()]


def prove_v43_export(udir: Path = UNITY_MOTION_V43, ref_dir: Path = INPUT_EXPORT, fit: str = "v43") -> dict:
    """the exported v43 motion against the input's export: the follow bin's sha256 = the input's (98657745...); the lead bin equal on every
    float except his right arm (collar, shoulder, elbow, wrist, fingers: their rotations and the positions of the shoulder and below) on the
    frames where the v43 fit differs from the input, and those frames inside the v43 windows (the adopted spans of curation_v43.json's
    stage) or the two despikes -> {"follow_sha_ok", "lead": {...}, "ok"}"""
    import hashlib
    from . import export_unity as eu
    out = {}
    fs = hashlib.sha256((udir / "smplx_follow.bin").read_bytes()).hexdigest()
    out["follow_sha256"] = fs
    out["follow_sha_ok"] = fs == INPUT_FOLLOW_SHA
    h1, m1 = eu.read_motion(udir / "smplx_lead.bin")
    h0, m0 = eu.read_motion(ref_dir / "smplx_lead.bin")
    T = len(m0)
    rot1, rot0 = m1[:, 3:3 + 55 * 4].reshape(T, 55, 4), m0[:, 3:3 + 55 * 4].reshape(T, 55, 4)
    pos1, pos0 = m1[:, 3 + 55 * 4:].reshape(T, 55, 3), m0[:, 3 + 55 * 4:].reshape(T, 55, 3)
    fi, f0 = load_fit(fit), load_fit("input")
    P1, P0 = hc.full_pose(fi["lead"]), hc.full_pose(f0["lead"])
    ch = np.abs(P1 - P0).max(2) > 0                                   # (T,55) joints the fit changed
    changed_frames = np.flatnonzero(ch.any(1))
    arm = np.array(V43_ARM_JOINTS)
    other = np.setdiff1d(np.arange(55), arm)
    # the descendants of his right collar: their positions follow the arm
    par = hc.Body(fi["lead"], "cpu").kin.parents
    desc = set()
    for j in range(55):
        k = j
        while k >= 0:
            if k == COLLAR["right"]:
                desc.add(j)
                break
            k = int(par[k])
    desc = np.array(sorted(desc))
    pos_other = np.setdiff1d(np.arange(55), desc)
    same_frames = np.setdiff1d(np.arange(T), changed_frames)
    lead = {"frames": int(T), "frames_changed_in_fit": int(len(changed_frames)),
            "joints_changed_in_fit": sorted(int(j) for j in np.flatnonzero(ch.any(0))),
            "max_abs_diff_unchanged_frames": float(np.abs(m1[same_frames] - m0[same_frames]).max()) if len(same_frames) else 0.0,
            "transl_max_abs_diff": float(np.abs(m1[:, :3] - m0[:, :3]).max()),
            "rotations_other_joints_max_abs_diff": float(np.abs(rot1[:, other] - rot0[:, other]).max()),
            "positions_not_below_the_collar_max_abs_diff": float(np.abs(pos1[:, pos_other] - pos0[:, pos_other]).max()),
            "header_equal_except_nothing": {k: (h1.get(k) == h0.get(k)) for k in set(h0) | set(h1)}}
    # the changed frames inside the windows (the adopted spans of the v43 stage) or the despikes
    allowed = np.zeros(T, bool)
    for a_s, b_s in V43_DESPIKE:
        allowed[int(round(a_s * HZ)):int(round(b_s * HZ)) + 1] = True
    for w in v43_kept_windows():
        allowed[w["span"][0]:w["span"][1] + 1] = True
    outside = np.setdiff1d(changed_frames, np.flatnonzero(allowed))
    lead["changed_frames_outside_windows"] = int(len(outside))
    lead["changed_frames_outside_windows_t"] = [round(x / HZ, 2) for x in outside[:20]]
    # fix round 1: his right collar against the input's, from the exported quaternions (the bound of PLAN H1: FIX1_COLLAR_BOUND_DEG)
    q1, q0 = rot1[:, COLLAR["right"]].astype(float), rot0[:, COLLAR["right"]].astype(float)
    dq = np.abs((q1 * q0).sum(1) / np.maximum(np.linalg.norm(q1, axis=1) * np.linalg.norm(q0, axis=1), 1e-12)).clip(0, 1)
    cang = np.degrees(2 * np.arccos(dq))
    lead["collar_change_vs_input_deg_max"] = round(float(cang.max()), 3)
    lead["collar_change_vs_input_frames_gt_bound"] = int((cang > FIX1_COLLAR_BOUND_DEG + 0.05).sum())
    out["lead"] = lead
    out["ok"] = bool(out["follow_sha_ok"] and lead["max_abs_diff_unchanged_frames"] == 0.0 and lead["transl_max_abs_diff"] == 0.0
                     and lead["rotations_other_joints_max_abs_diff"] == 0.0 and lead["positions_not_below_the_collar_max_abs_diff"] == 0.0
                     and set(lead["joints_changed_in_fit"]) <= set(int(j) for j in arm) and lead["changed_frames_outside_windows"] == 0
                     and lead["collar_change_vs_input_frames_gt_bound"] == 0)
    return out


def _write_install_script_v43(dst: Path | None = None) -> Path:
    """v4/contacts/unity_motion_v43/install_motion.sh: the v4 install script with this folder as the source; its backup (made by its first
    install, never overwritten) holds the motion installed then - v4.2's 8bd1d1ad - so --restore returns v4.2 (also kept in
    ../unity_motion_installed_8bd1/). Fix round 1: the source = the script's own folder, the capture overridable (HM_CAPTURE_DIR, the
    test's temporary capture), and the recorded sha256 read from the manifest's "sha256" object only (the first match)"""
    src = (UNITY_MOTION / "install_motion.sh").read_text(encoding="utf-8") if (UNITY_MOTION / "install_motion.sh").exists() else None
    if src is None:
        src = (OUT / "unity_motion_installed_8bd1" / "install_motion.sh").read_text(encoding="utf-8")
    s = src.replace('SRC="$WS/dancecap/work/larissa_kadu_recap/v4/contacts/unity_motion"',
                    'SRC="$(cd "$(dirname "$0")" && pwd)"                         # this folder (unity_motion_v43)')
    s = s.replace('CAP="$WS/head-movement/Assets/StreamingAssets/06_LarissaKaduRecap"',
                  'CAP="${HM_CAPTURE_DIR:-$WS/head-movement/Assets/StreamingAssets/06_LarissaKaduRecap}"')
    s = s.replace("""want="$(grep -o "\\"$f\\": \\"[0-9a-f]\\{64\\}\\"" "$MAN" | grep -o '[0-9a-f]\\{64\\}' || true)\"""",
                  """want="$(grep -o "\\"$f\\": \\"[0-9a-f]\\{64\\}\\"" "$MAN" | head -1 | grep -o '[0-9a-f]\\{64\\}' || true)\"""")
    if "| head -1 |" not in s or "HM_CAPTURE_DIR" not in s or 'SRC="$(cd' not in s:
        raise RuntimeError("install_motion.sh v43: the source script changed; the fix-round-1 replacements did not apply")
    s = s.replace("put the v4 zouk-contact motion into the installed recap capture.",
                  "put the v43 motion (the detected poses + his right hand on her upper-left back) into the installed recap capture.")
    s = s.replace("dancecap/work/larissa_kadu_recap/v4/contacts/unity_motion/install_motion.sh",
                  "dancecap/work/larissa_kadu_recap/v4/contacts/unity_motion_v43/install_motion.sh")
    s = s.replace("(run: python -m dancecap.zouk_contacts unity-motion)", "(run: python -m dancecap.zouk_contacts unity-motion --fit v43)")
    s = s.replace('echo "installed the v4 zouk-contact motion (dancecap.zouk_contacts) into $CAP ($FILES)"',
                  'echo "installed the v43 motion (dancecap.zouk_contacts, section v43) into $CAP ($FILES)"')
    s = s.replace("# put the backed-up (installed before v4) files back", "# put the backed-up files (the motion installed before v43: v4.2) back")
    head_old = s[s.index("# Version:"):s.index("# Only the four motion files change")]
    head_new = ("# Version: v43 (dancecap.zouk_contacts, section v43, 2026-10-23): both dancers = the input fit (smplx_rf2_noslip) except his\n"
                "# right arm in the torso-hold windows (decisions in v4/contacts/curation_v43.json + gate_<tag>.json, windows plan_v43.json),\n"
                "# numbers in v4/contacts/validate_v43.md, the export, its proof against the input's export and the sha256 of each file in\n"
                "# v4/contacts/unity_motion_v43/unity_motion.json. The v4.2 motion (smplx_lead.bin 8bd1d1ad...) is also kept in\n"
                "# v4/contacts/unity_motion_installed_8bd1/ (its fit in v4/smplx_contact_installed_8bd1/).\n")
    s = s.replace(head_old, head_new)
    i0 = s.index("if [ ! -f \"$BAK/smplx_lead.bin\" ]; then")
    i1 = s.index("  for f in $FILES capture.json;", i0)
    s = s[:i0] + ("if [ ! -f \"$BAK/smplx_lead.bin\" ]; then                       # never overwrite the first backup: the motion installed "
                  "before v43\n                                                              # (v4.2's 8bd1d1ad...); --restore puts it back\n") + s[i1:]
    p = Path(dst or UNITY_MOTION_V43) / "install_motion.sh"
    p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    return p


def back_phone_sees(fit: dict, obs, hcfg: HoldCfg, back_fv: float = 0.1, min_lm: int = 6) -> np.ndarray:
    """(T,) a phone that sees her back (her front away from it) sees his right hand (>= min_lm MAMMA hand points, not swapped with his left)
    - the frames where the phones can support where his hand lies on her back"""
    T = len(fit["lead"]["t"])
    ks = np.arange(T)
    Jf = posed(fit["follow"], ks)["J"]
    fwd = torso_frame(Jf)[:, :, 2]
    out = np.zeros(T, bool)
    for a in range(0, T, 2000):
        kk = ks[a:a + 2000]
        PL = hc.full_pose(fit["lead"])
        hr = hc.hand_points_np(fit["lead"], PL, kk, "right")["lm"]
        hl = hc.hand_points_np(fit["lead"], PL, kk, "left")["lm"]
        for c in obs.cams:
            z = obs.z[c]
            Cc = -np.einsum("kji,kj->ki", z["R"][kk], z["t"][kk]) - Jf[kk, SPINE3]
            Cc[:, 1] = 0
            fv = (Cc * fwd[kk]).sum(1) / np.maximum(np.linalg.norm(Cc, axis=1), 1e-9)
            see, _ = hand_2d(obs, c, kk, HoldCfg(min_lm=min_lm))
            out[kk] |= see & (fv < back_fv) & ~mamma_swapped(obs, c, kk, [hr], hl, hcfg)
    return out


def palm_location(fit: dict, sites_f: dict, cls: np.ndarray, lift_m: float = 0.015, near_m: float = 0.08) -> dict:
    """per frame where his right palm is on her (her torso frame): "ulb" within near_m of her upper-left back (ULB5; the counterbalance class
    also the top of her left shoulder), "back" (z < -6 cm), "front" (z > +4 cm), "flank" (else) -> {"loc" (T,) str, "z_m", "d_ulb_m"}"""
    T = len(fit["lead"]["t"])
    ks = np.arange(T)
    kl, kf = posed(fit["lead"], ks), posed(fit["follow"], ks)
    pc = hand_site(kl["J"], "right", "palm")[0]
    pts = v43_site_points(kf, sites_f, lift_m, ULB5 + ("shoulder_top_L",))
    d_ulb = np.min([np.linalg.norm(pts[s][0] - pc, axis=1) for s in ULB5], 0)
    d_cb = np.minimum(d_ulb, np.linalg.norm(pts["shoulder_top_L"][0] - pc, axis=1))
    z = in_frame(pc, kf["J"])[:, 2]
    loc = np.where(z > 0.04, "front", np.where(z < -0.06, "back", "flank")).astype(object)
    on = np.where(np.asarray(cls) == "cb", d_cb, d_ulb) <= near_m
    loc[on & (z <= 0.04)] = "ulb"
    return {"loc": loc.astype(str), "z_m": z, "d_ulb_m": d_ulb}


def validate_v43(fits=("input", "v3", "v4", "v43"), path: Path | None = None) -> dict:
    """the v43 tables over the whole film and at the owner's three times (OWNER_WINDOWS_S): per hold where his right palm is (her upper-left
    back / other back / flank / front), the share on her upper-left back where a back-view phone sees his hand, inside her (exact: torso /
    shoulders / upper arms > 2 / > 5 cm), the wrist (bend, twist) and elbow, the arm's added speed against the input, pops at the window edges,
    and that the hand holds and her whole motion are the input exactly -> validate_v43.json + validate_v43.md"""
    t0 = time.time()
    H = holds_with_overrides(load_holds(), V43_HOLD_OVERRIDES)
    T = len(H["names"])
    cbi = counterbalance_intervals()
    cls = v43_classes(H["names"], np.arange(T), cbi)
    fl = {f: load_fit(f) for f in fits}
    sites = site_bodies(fl["input"])
    obs = hc.Obs()
    hcfg = HoldCfg()
    sees = back_phone_sees(fl["input"], obs, hcfg)
    holds = ("upper_back", "shoulder_top", "wrap_back", "behind_back")
    rep = {"what": "v43 validation: the leader's right hand (input = smplx_rf2_noslip, v3, v4 = the v4.2 motion 8bd1d1ad, v43)", "frames": T,
           "holds_frames": {h: int((H["names"] == h).sum()) for h in holds}, "classes_frames": {c: int((cls == c).sum()) for c in ("ulb", "cb", "back")},
           "back_phone_sees_hand_frames": int(sees.sum()), "fits": {}, "owner_windows_s": OWNER_WINDOWS_S}
    masks = {"all torso holds": np.isin(H["names"], holds)}
    for h in holds:
        masks[h] = H["names"] == h
    masks["counterbalance (cb class)"] = cls == "cb"
    for i, (a, b) in enumerate(OWNER_WINDOWS_S):
        tt = np.arange(T) / HZ
        masks[f"owner {['1:30', '2:30', '4:50'][i]} (dance {a}-{b})"] = (tt >= a) & (tt <= b) & np.isin(H["names"], holds)
    P_in = {r: hc.full_pose(fl["input"][r]) for r in ROLES}
    for f, fit in fl.items():
        r = {}
        loc = palm_location(fit, sites["follow"], cls)
        with np.load(OUT / f"exact_{f}.npz") as z:
            parts = [str(x) for x in z["parts"]]
            pen = z["hand_pen_part"][:, [parts.index(p) for p in V43_EXACT_PARTS]].max(1)
            arm_pen = z["arm_pen"].copy()
            gap = z["hand_gap"].copy()
        Wa, Ea = wrist_angles(hc.full_pose(fit["lead"]), "right"), elbow_angles(hc.full_pose(fit["lead"]), "right")
        per = {}
        for nm, m in masks.items():
            n = int(m.sum())
            if not n:
                continue
            L = loc["loc"][m]
            ms = m & sees
            per[nm] = {"frames": n, "ulb": round(float((L == "ulb").mean()), 3), "back": round(float((L == "back").mean()), 3),
                       "flank": round(float((L == "flank").mean()), 3), "front": round(float((L == "front").mean()), 3),
                       "ulb_where_back_phone_sees": round(float((loc["loc"][ms] == "ulb").mean()), 3) if ms.any() else None,
                       "on_her_back_or_ulb_where_back_phone_sees": round(float(np.isin(loc["loc"][ms], ("ulb", "back")).mean()), 3) if ms.any() else None,
                       "frames_back_phone_sees": int(ms.sum()),
                       "inside_gt2cm": int((pen[m] > 0.02).sum()), "inside_gt5cm": int((pen[m] > 0.05).sum()),
                       "forearm_inside_gt2cm": int((arm_pen[m] > 0.02).sum()), "hand_off_her_gt2cm": int((gap[m] > 0.02).sum()),
                       "bend_gt85": int((Wa["bend"][m] > 85).sum()), "bend_gt120": int((Wa["bend"][m] > 120).sum()),
                       "twist_gt120": int((np.abs(Wa["twist"][m]) > 120).sum()), "elbow_hyper_lt_m5": int((Ea["flexion"][m] < -5).sum())}
        r["per_mask"] = per
        r["whole_film"] = {"inside_gt2cm": int((pen > 0.02).sum()), "inside_gt5cm": int((pen > 0.05).sum()),
                           "bend_gt120": int((Wa["bend"] > 120).sum()), "bend_gt150": int((Wa["bend"] > 150).sum()),
                           "twist_gt120_bend_lt60": int(((np.abs(Wa["twist"]) > 120) & (Wa["bend"] < 60)).sum()),
                           "elbow_hyper_lt_m5": int((Ea["flexion"] < -5).sum()), "elbow_off_hinge_gt35": int((np.abs(Ea["off_hinge"]) > 35).sum())}
        if f != "input":
            r["arm_speed_vs_input"] = arm_speed_checks(fl["input"], fit)
            # the hand holds and her motion: the input exactly?
            PL = hc.full_pose(fit["lead"])
            left = [13, 16, 18, 20, *range(25, 40)]
            r["follow_max_abs_diff_vs_input"] = float(np.abs(hc.full_pose(fit["follow"]) - P_in["follow"]).max())
            r["follow_transl_max_abs_diff"] = float(np.abs(np.asarray(fit["follow"]["transl"]) - np.asarray(fl["input"]["follow"]["transl"])).max())
            r["lead_left_arm_max_abs_diff_vs_input"] = float(np.abs(PL[:, left] - P_in["lead"][:, left]).max())
            other = np.setdiff1d(np.arange(55), np.array(V43_ARM_JOINTS))
            r["lead_not_right_arm_max_abs_diff_vs_input"] = float(np.abs(PL[:, other] - P_in["lead"][:, other]).max())
            r["lead_transl_max_abs_diff_vs_input"] = float(np.abs(np.asarray(fit["lead"]["transl"]) - np.asarray(fl["input"]["lead"]["transl"])).max())
            hh = H["names"] == "hand_hold"
            r["lead_right_arm_max_abs_diff_on_right_hand_holds"] = float(np.abs(PL[hh][:, list(V43_ARM_JOINTS)] - P_in["lead"][hh][:, list(V43_ARM_JOINTS)]).max()) if hh.any() else 0.0
            r["frames_changed_vs_input"] = int((np.abs(PL - P_in["lead"]).max((1, 2)) > 0).sum())
        rep["fits"][f] = r
        log(f"validate_v43 {f} ({time.time() - t0:.0f} s)")
    if "v43" in fl:
        rep["window_edges"] = window_edge_speeds(fl["input"], fl["v43"])
    write_json(OUT / "validate_v43.json", rep)
    (path or (OUT / "validate_v43.md")).write_text(validate_v43_md(rep), encoding="utf-8")
    return rep


def window_edge_speeds(fit_in: dict, fit: dict, tag: str | None = None, around: int = 6) -> dict:
    """at every edge of every kept v43 window span (curation_v43.json's stages): the largest ADDED speed (the change against the input, m/s) of
    his right elbow, wrist, palm centre and fingertips within +-around frames, and the speed against max(input p99, the input there) + 0.25"""
    cur = json.loads(curation_file("v43").read_text(encoding="utf-8")) if curation_file("v43").exists() else {}
    P_in, P = hc.full_pose(fit_in["lead"]), hc.full_pose(fit["lead"])
    oi, o1 = posed(fit_in["lead"], None, P_in), posed(fit["lead"], None, P)
    pts = {"elbow": (oi["J"][:, ELBOW["right"]], o1["J"][:, ELBOW["right"]]), "wrist": (oi["J"][:, WRIST["right"]], o1["J"][:, WRIST["right"]]),
           "palm": (hc.palm_centre(oi["J"], "right"), hc.palm_centre(o1["J"], "right"))}
    for i, fi in enumerate(FINGERS):
        pts[f"tip_{fi}"] = (oi["tips"][:, tip_index("right", fi)], o1["tips"][:, tip_index("right", fi)])
    add = {k: hc.speeds(x1 - xi) for k, (xi, x1) in pts.items()}
    fast = {k: hc.speeds(x1) - np.maximum(np.percentile(hc.speeds(xi), 99), hc.speeds(xi)) for k, (xi, x1) in pts.items()}
    # the audit's whip: the added speed over 0.55 m/s WHILE the point moves faster than the input (a calm hand resting on her while the
    # input's flails adds speed against it without any pop)
    addf = {k: np.where(hc.speeds(x1) > hc.speeds(xi), hc.speeds(x1 - xi), 0.0) for k, (xi, x1) in pts.items()}
    T = len(P)
    out = {"edges": 0, "max_added_mps": 0.0, "edges_added_gt_0.55": 0, "edges_added_gt_0.55_and_faster": 0, "edges_fast": 0, "worst": []}
    for w in v43_kept_windows(cur):
        for e in w["span"]:
            sl = slice(max(0, e - around), min(T - 1, e + around))
            a = max(float(v[sl].max()) for v in add.values())
            af = max(float(v[sl].max()) for v in addf.values())
            fs = max(float(v[sl].max()) for v in fast.values())
            out["edges"] += 1
            out["max_added_mps"] = max(out["max_added_mps"], round(a, 3))
            out["edges_added_gt_0.55"] += int(a > 0.55)
            out["edges_added_gt_0.55_and_faster"] += int(af > 0.55)
            out["edges_fast"] += int(fs > 0.25)
            out["worst"].append((round(af, 3), round(a, 3), round(e / HZ, 2), w["id"]))
    out["worst"] = sorted(out["worst"], reverse=True)[:8]
    allf = np.max(np.stack(list(addf.values())), 0)
    out["film_frames_added_gt_0.55_and_faster"] = int((allf > 0.55).sum())
    out["film_frames_added_gt_0.55"] = int((np.max(np.stack(list(add.values())), 0) > 0.55).sum())
    return out


def validate_v43_md(rep: dict) -> str:
    fits = list(rep["fits"])
    lab = {"v4": "v4 (v4.2, 8bd1d1ad)"}
    out = ["# Leader's right hand, v43: where it is on her (" + " | ".join(lab.get(f, f) for f in fits) + ")", "",
           f"{rep['frames']} frames at 30 Hz. Holds (review mislabels overridden: 49.6-53.8 s hand_hold, 130-142 s free): "
           + ", ".join(f"{h} {n}" for h, n in rep["holds_frames"].items()) + f". A back-view phone sees his right hand on "
           f"{rep['back_phone_sees_hand_frames']} frames. ULB = his palm within 8 cm of her upper-left back (UPPER_LEFT_BACK + shoulder_blade_L; "
           "in the counterbalance leans also the top of her left shoulder); back = z < -6 cm in her torso frame; front = z > +4 cm. Inside = exact "
           "depth into her torso / shoulders / upper arms.", ""]
    keys = [("ulb", "palm on her upper-left back (share)"), ("ulb_where_back_phone_sees", "... where a back-view phone sees his hand"),
            ("on_her_back_or_ulb_where_back_phone_sees", "on her back (ULB or other) where a back-view phone sees it"),
            ("back", "other back (share)"), ("flank", "flank (share)"), ("front", "her front (share)"),
            ("inside_gt2cm", "inside her > 2 cm (frames)"), ("inside_gt5cm", "inside her > 5 cm (frames)"),
            ("forearm_inside_gt2cm", "his forearm inside her (any part) > 2 cm (frames)"), ("hand_off_her_gt2cm", "hand > 2 cm off her (frames)"),
            ("bend_gt85", "wrist bent > 85 deg (frames)"), ("bend_gt120", "wrist folded > 120 deg (frames)"), ("twist_gt120", "wrist twist > 120 deg"),
            ("elbow_hyper_lt_m5", "elbow hyperextended > 5 deg")]
    masks = list(rep["fits"][fits[0]]["per_mask"])
    for nm in masks:
        n = rep["fits"][fits[0]]["per_mask"][nm]["frames"]
        nb = rep["fits"][fits[0]]["per_mask"][nm]["frames_back_phone_sees"]
        out += [f"## {nm} ({n} frames; a back-view phone sees the hand on {nb})", "", "| measure | " + " | ".join(lab.get(f, f) for f in fits) + " |",
                "|---|" + "---|" * len(fits)]
        for k, t in keys:
            out.append(f"| {t} | " + " | ".join(str(rep["fits"][f]["per_mask"].get(nm, {}).get(k, "-")) for f in fits) + " |")
        out.append("")
    out += ["## Whole film", "", "| measure | " + " | ".join(lab.get(f, f) for f in fits) + " |", "|---|" + "---|" * len(fits)]
    for k in rep["fits"][fits[0]]["whole_film"]:
        out.append(f"| {k} | " + " | ".join(str(rep["fits"][f]["whole_film"][k]) for f in fits) + " |")
    sp = [f for f in fits if "arm_speed_vs_input" in rep["fits"][f]]
    for k, t in (("elbow", "elbow"), ("wrist", "wrist"), ("palm", "palm centre")):
        out.append(f"| added speed vs the input > 0.55 m/s, {t} (frames) / faster than max(p99, input) + 0.25 | - | "
                   + " | ".join(f"{rep['fits'][f]['arm_speed_vs_input'][k]['added_gt_0.55']} / {rep['fits'][f]['arm_speed_vs_input'][k]['fast']}"
                                for f in sp) + " |")
    out.append("| added speed > 0.55 m/s, fingertips (worst finger, frames) | - | "
               + " | ".join(str(rep["fits"][f]["arm_speed_vs_input"]["fingertips_added_gt_0.55"]) for f in sp) + " |")
    out.append("| wrist turning > 8 deg / frame faster than the input (frames) | - | "
               + " | ".join(str(rep["fits"][f]["arm_speed_vs_input"]["wrist_rotation_faster_than_input_8deg"]) for f in sp) + " |")
    for k, t in (("follow_max_abs_diff_vs_input", "her poses vs the input, max abs diff (rad)"),
                 ("follow_transl_max_abs_diff", "her root translation vs the input (m)"),
                 ("lead_left_arm_max_abs_diff_vs_input", "his left arm + hand (the hand holds) vs the input (rad)"),
                 ("lead_right_arm_max_abs_diff_on_right_hand_holds", "his right arm on the frames his right hand holds hers vs the input (rad)"),
                 ("lead_not_right_arm_max_abs_diff_vs_input", "his body except the right arm vs the input (rad)"),
                 ("lead_transl_max_abs_diff_vs_input", "his root translation vs the input (m)"),
                 ("frames_changed_vs_input", "his frames that differ from the input")):
        out.append(f"| {t} | - | " + " | ".join(str(rep["fits"][f].get(k, "-")) for f in sp) + " |")
    if rep.get("window_edges"):
        we = rep["window_edges"]
        out += ["", f"Window edges (v43, {we['edges']} edges of the kept window spans, +-6 frames): the largest added speed of his right elbow / "
                    f"wrist / palm / fingertips against the input {we['max_added_mps']} m/s; edges over 0.55 m/s: {we['edges_added_gt_0.55']}, of "
                    f"them while moving faster than the input (the audit's whip): {we['edges_added_gt_0.55_and_faster']}; edges faster than "
                    f"max(p99, input) + 0.25: {we['edges_fast']}. Whole film: frames with an added speed > 0.55 m/s "
                    f"{we['film_frames_added_gt_0.55']}, of them faster than the input {we['film_frames_added_gt_0.55_and_faster']}. "
                    f"Worst (added while faster, added, s, window): {we['worst']}"]
    return "\n".join(out) + "\n"


def make_input_run(tag: str = "input", fit: str | None = None) -> Path:
    """runs/<tag>/poses_merged.npz = the lead poses of the fit `fit` (default: tag; "input" for the first v43 round) - the base motion of a
    v43 curation stage (apply_curation base_tag)"""
    d = OUT / "runs" / tag
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "poses_merged.npz", lead=hc.full_pose(load_fit(fit or tag)["lead"]))
    return d / "poses_merged.npz"


def freeze_v43(name: str = "v43a") -> dict:
    """the curated v43 fit (v4/smplx_contact_v43) frozen as the next round's base: v4/smplx_contact_<name>, exact_<name>.npz, measure_<name>.npz
    (copies) and runs/<name>/poses_merged.npz"""
    import shutil
    dst = FITS[name]
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(FIT_V43, dst)
    for k in ("exact", "measure"):
        if (OUT / f"{k}_v43.npz").exists():
            shutil.copy2(OUT / f"{k}_v43.npz", OUT / f"{k}_{name}.npz")
    p = make_input_run(name, name)
    cp = curation_file("v43")
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        c.setdefault("frozen_stages", {})[name] = list(c.get("fit_stages") or [])
        cp.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"fit": str(dst), "poses": str(p)}


def v43_kept_windows(cur: dict | None = None) -> list[dict]:
    """the windows the written v43 fit carries: the adopted, not vetoed windows of the stages in curation_v43.json["fit_stages"]"""
    cur = cur if cur is not None else (json.loads(curation_file("v43").read_text(encoding="utf-8")) if curation_file("v43").exists() else {})
    out = []
    for tg in cur.get("fit_stages") or []:
        st = (cur.get("stages") or {}).get(tg) or {}
        vet = set((st.get("veto") or {}).keys())
        out += [w for w in adopted_windows([tg]) if w["id"] not in vet]
    return out


# ------------------------------------------------------------------------------------------------ v43 fix round 1 (2026-10-23)
#
# The independent check of v43 (verify H round 0) found: his right collar unbounded over the rounds (up to 56 deg against the input, his
# shoulder dropped up to 11 cm), the counterbalance arm at 1:30 a "V" hooked over the top of her left shoulder from the front, the first
# counterbalance (54-65 s) and the 21-s opening closed hold (28-50 s) unchanged on her front, 2:30 worse than the input, his hand in her neck at
# 92.6 / 152.5 / 240.5 s, small speed breaches, and the v43 install script refusing to install. The fix:
#   collar    the collar's rotation against the INPUT's collar saturated to FIX1_COLLAR_BOUND_DEG (a soft knee, C1, <= 2 deg / frame), then
#             the shoulder / elbow / wrist re-solved so that the wrist and the hand stay where they were (arm_keep_hand: the elbow turned only
#             about its hinge, the arm turned by its smallest rotation and swivelled to the elbow nearest where it was, the hand's global
#             rotation kept exactly - no wrist-twist cap in the collar fix: a cap there snapped the fingertips);
#   re-pose   the problem runs (the palm on her front / off her upper-left back / the elbow sagging in a counterbalance lean / the hand in her
#             neck), in chunks of <= 8 s (3-s ramps), re-posed ANALYTICALLY (fix1_window): candidate targets ON HER EXPORTED SKIN (SkinLBS: the
#             shape-edited mesh Unity shows) - ray exits from her spine axis round her left side to her back (never over the top of her
#             shoulder), the back of her left shoulder, the input's palm where it was on her upper-left back, the base palm where it is fine
#             - with a location cost per class, the arm IK (collar budget counted against the input), reach / wrist / COCO wrist costs and
#             his skin's hand / forearm depth in her at the target; a Viterbi; the track smoothed and rate-limited in her frame, its ramps
#             in her cylindrical coordinates (round her side, never through her); the elbow's swivel by his forearm's depth in her, his hand
#             in her neck, the COCO elbow, the counterbalance sag, the wrist twist; unreachable frames laid onto her surface where the arm
#             reaches; the hand tilted / lifted where its skin sinks into hers; the blend from the base in the target's terms (the hand's
#             facing by a rotation slerp, never a vector blend that turns it edge-on into her); the hand pushed out of her along the
#             deepest point's way out (hand_way_out; her neck: away from its axis); the speed rules repaired. FIX1_VARIANTS in turn until
#             fix1_check passes (location by class; hand / forearm depth, neck, sag, speed, wrist / elbow limits, COCO elbow / wrist with
#             the phones that see the candidate's joint - against the base and the input); a base with his hand in her neck that no
#             re-pose fixes: the base with only the hand moved out (fix1_pushout); else the window keeps the base; a vetoed window next to
#             a kept one once more (its ramp then starts from the kept motion);
#   output    the kept windows on the parts' base, the current collar fix outside them, the whips outside them low-passed with a depth
#             guard (fix1_whips; no global speed repair: it put his hand into her), the collar bounded again (collar_reclamp), the stage
#             "fix1" of curation_v43.json (runs/fix1/) on the pre-fix v43 frozen as v43f, the fit v43, its export with the max collar
#             change against the input in the proof. On a pod: fix1 --base v43f --part i/n in parallel (independent window groups),
#             fix1-merge --base v43f --write (cheap: may run locally on the pods' parts).

FIX1_STAGE = "fix1"
FIX1_COLLAR_BOUND_DEG = 15.0           # his right collar never turns more than this against the input's collar (hard) ...
FIX1_COLLAR_KNEE_DEG = 10.0            # ... saturating smoothly above this (C1)
SKINS_V43 = OUT / "skins_v43"           # the exported (shape-edited) skins, copied from the v43 side export: the geometry Unity shows


def saturate_deg(a: np.ndarray, bound: float = FIX1_COLLAR_BOUND_DEG, knee: float = FIX1_COLLAR_KNEE_DEG) -> np.ndarray:
    """angles (deg) kept below knee, compressed above it (knee + (bound - knee) tanh(...)): monotone, C1, never above bound"""
    a = np.asarray(a, float)
    w = max(float(bound) - float(knee), 1e-9)
    return np.where(a <= knee, a, knee + w * np.tanh((a - knee) / w))


def collar_bounded(P_in: np.ndarray, P: np.ndarray, bound: float = FIX1_COLLAR_BOUND_DEG, knee: float = FIX1_COLLAR_KNEE_DEG,
                   joint: int | None = None, step_deg: float | None = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """the poses P (T,55,3) with his right collar's rotation against the input's (P_in) saturated (saturate_deg of its angle, the same axis)
    and its change from frame to frame limited to step_deg (limit_steps; frames whose collar equals the input's never change)
    -> (P', angle before (deg, T)); frames whose collar equals the input's stay bit-identical"""
    from scipy.spatial.transform import Rotation as Ro
    j = COLLAR["right"] if joint is None else int(joint)
    P = np.asarray(P, float).copy()
    Ri, Rc = Ro.from_rotvec(P_in[:, j]), Ro.from_rotvec(P[:, j])
    rel = (Ri.inv() * Rc).as_rotvec()
    a = np.degrees(np.linalg.norm(rel, axis=1))
    changed = np.abs(P[:, j] - P_in[:, j]).max(1) > 0
    s = np.where(a > knee, saturate_deg(a, bound, knee) / np.maximum(a, 1e-12), 1.0)
    rel2 = rel * s[:, None]
    if step_deg is not None and step_deg > 0:
        rel2 = limit_steps(rel2, np.radians(step_deg), fixed=~changed)
        n2 = np.degrees(np.linalg.norm(rel2, axis=1))
        rel2 = rel2 * np.where(n2 > bound, bound / np.maximum(n2, 1e-12), 1.0)[:, None]
    m = changed & (np.abs(rel2 - rel).max(1) > 0)
    if m.any():
        P[m, j] = (Ri[m] * Ro.from_rotvec(rel2[m])).as_rotvec()
    return P, a


def limit_steps(X: np.ndarray, step: float, fixed: np.ndarray | None = None, iters: int = 400) -> np.ndarray:
    """a track X (T,d) whose frame-to-frame change is brought to at most `step` by local averaging (0.25 / 0.5 / 0.25) of the two frames of
    each faster step; frames in `fixed` never change (a step into a fixed frame is spread over the free side only)"""
    X = np.asarray(X, float).copy()
    T = len(X)
    fixed = np.zeros(T, bool) if fixed is None else np.asarray(fixed, bool)
    for _ in range(iters):
        v = np.linalg.norm(np.diff(X, axis=0), axis=1)
        bad = np.flatnonzero(v > step * 1.0001)
        if not len(bad):
            break
        idx = np.unique(np.concatenate([bad, bad + 1]))
        idx = idx[~fixed[idx]]
        if not len(idx):
            break
        lo, hi = np.maximum(idx - 1, 0), np.minimum(idx + 1, T - 1)
        X[idx] = 0.25 * X[lo] + 0.5 * X[idx] + 0.25 * X[hi]
    return X


def min_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(W,3,3) the smallest rotations taking the directions a (W,3) onto b (W,3)"""
    a, b = _unit(np.asarray(a, float)), _unit(np.asarray(b, float))
    c = np.cross(a, b)
    s = np.linalg.norm(c, axis=1)
    d = (a * b).sum(1)
    ang = np.arctan2(s, d)
    ax = np.where(s[:, None] > 1e-12, c / np.maximum(s, 1e-12)[:, None], _unit(np.cross(a, np.array([0.0, 0.0, 1.0])),
                                                                             _unit(np.cross(a, np.array([1.0, 0.0, 0.0])))))
    return _rot_about(ax, ang)


def elbow_for_distance(R19: np.ndarray, r19: np.ndarray, r21: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(W,) the angle (rad) to turn the elbow's local rotation R19 (W,3,3) about the hinge axis (the upper arm's local y: elbow_angles' flexion
    plane; the flexion grows by delta) so that the shoulder-wrist distance |r19 + Ry(delta) R19 r21| becomes dist (W,): of the two mirror
    solutions the BENT one (the larger flexion; the other bends the elbow backwards); clamped at the straightest / most bent reachable"""
    y = np.array([0.0, 1.0, 0.0])
    v = np.einsum("wij,j->wi", R19, r21)
    vy = v[:, 1]
    vp = v.copy()
    vp[:, 1] = 0.0
    rp = np.asarray(r19, float).copy()
    ry = rp[1]
    rp[1] = 0.0
    a_, b_ = np.linalg.norm(rp), np.linalg.norm(vp, axis=1)
    const = float(r19 @ r19) + (v * v).sum(1) + 2 * ry * vy
    c = (np.asarray(dist, float) ** 2 - const) / np.maximum(2 * a_ * b_, 1e-12)        # = cos(phi0 + delta)
    # phi0: the signed angle (about +y) from rp to vp
    phi0 = np.arctan2(np.cross(np.tile(rp, (len(v), 1)), vp) @ y, vp @ rp)
    base = np.arccos(np.clip(c, -1.0, 1.0))
    cands = np.stack([base - phi0, -base - phi0], 1)
    cands = (cands + np.pi) % (2 * np.pi) - np.pi
    a = _rest_axes("right")[0]
    h = np.einsum("wij,j->wi", R19, a)
    f = np.arctan2(h[:, 2], h @ a)                                  # elbow_angles' flexion now
    fl = (f[:, None] + cands + np.pi) % (2 * np.pi) - np.pi
    return cands[np.arange(len(v)), np.argmax(fl, 1)]


def arm_keep_hand(fit_lead: dict, P_ref: np.ndarray, P_new: np.ndarray, ks: np.ndarray, X_wrist: np.ndarray | None = None,
                  E_hint: np.ndarray | None = None, G21: np.ndarray | None = None, twist_cap_deg: float = 120.0) -> tuple[np.ndarray, dict]:
    """his right arm in P_new (the frames ks; the collar already changed against P_ref) re-solved so that the wrist and the hand stay where
    P_ref has them (or the wrist at X_wrist (n,3), the hand's global rotation G21 (n,3,3)): the elbow keeps its hinge (its local rotation
    turned only about the hinge axis: no new off-hinge, no new forearm roll) to reach the wrist's distance, the arm turned about the shoulder
    by the smallest rotation onto the wrist and swivelled about the shoulder-wrist axis to the elbow nearest P_ref's elbow moved with the
    shoulder (or E_hint), the hand's global rotation kept (beyond the arm's reach the wrist comes as close as it can)
    -> (P_new with joints 17 / 19 / 21 set on ks, {"palm_moved_m", "elbow_moved_m", "wrist_miss_m"})"""
    from scipy.spatial.transform import Rotation as Ro
    P_new = np.asarray(P_new, float).copy()
    ks = np.asarray(ks)
    if not len(ks):
        return P_new, {"palm_moved_m": np.zeros(0), "elbow_moved_m": np.zeros(0), "wrist_miss_m": np.zeros(0)}
    rest = arm_rest(fit_lead["betas"])
    r19, r21 = np.asarray(rest["r19"], float), np.asarray(rest["r21"], float)
    o0 = posed(fit_lead, ks, P_ref[ks])
    o1 = posed(fit_lead, ks, P_new[ks])
    sh, el, wr, co = SHOULDER["right"], ELBOW["right"], WRIST["right"], COLLAR["right"]
    S0, E0, W0 = o0["J"][:, sh], o0["J"][:, el], o0["J"][:, wr]
    S1 = o1["J"][:, sh]
    G14 = o1["R"][:, co]
    G17o, G21o = o0["R"][:, sh], (o0["R"][:, wr] if G21 is None else np.asarray(G21, float))
    R19o = rotvec_to_mat(P_ref[ks, el])
    Wg = W0 if X_wrist is None else np.asarray(X_wrist, float)
    D = Wg - S1
    d = np.linalg.norm(D, axis=1)
    u = _unit(D)
    dlt = elbow_for_distance(R19o, r19, r21, d)
    R19n = _rot_about(np.tile([0.0, 1.0, 0.0], (len(ks), 1)), dlt) @ R19o
    Wl = r19 + np.einsum("wij,j->wi", R19n, r21)                   # the wrist in the upper arm's frame
    Wtmp = np.einsum("wij,wj->wi", G17o, Wl)
    Rm = min_rot(Wtmp, D)
    G17a = Rm @ G17o
    Eh = (E0 + (S1 - S0)) if E_hint is None else np.asarray(E_hint, float)
    ea = np.einsum("wij,j->wi", G17a, r19)
    eh = Eh - S1
    pa = ea - (ea * u).sum(1, keepdims=True) * u
    ph = eh - (eh * u).sum(1, keepdims=True) * u
    psi = np.arctan2((np.cross(pa, ph) * u).sum(1), (pa * ph).sum(1))
    psi = np.where((np.linalg.norm(pa, axis=1) > 1e-6) & (np.linalg.norm(ph, axis=1) > 1e-6), psi, 0.0)
    G17 = _rot_about(u, psi) @ G17a
    G19 = G17 @ R19n
    R17 = np.transpose(G14, (0, 2, 1)) @ G17
    R21 = np.transpose(G19, (0, 2, 1)) @ G21o
    # where keeping the hand's global rotation would twist the wrist past max(twist_cap_deg, its twist before + 5) the hand turns part of
    # the way with the forearm (its local rotation before), just enough - the share smoothed over consecutive frames (a share switching on
    # and off from frame to frame whips the fingertips: 1.4-2.3 m/s at 92.2-92.5 s with an 85-deg cap in fix round 1; 120 caps only the
    # candy-wrapper twists)
    R21o = rotvec_to_mat(P_ref[ks, wr])
    tw0 = np.abs(wrist_angles_R(R21o, "right")["twist"])
    lim = np.maximum(twist_cap_deg, tw0 + 5.0)
    bad = np.abs(wrist_angles_R(R21, "right")["twist"]) > lim
    if bad.any():
        from scipy.ndimage import gaussian_filter1d, maximum_filter1d
        qa, qb = Ro.from_matrix(R21).as_rotvec(), Ro.from_matrix(R21o).as_rotvec()
        lo, hi = np.zeros(bad.sum()), np.ones(bad.sum())
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            okm = np.abs(wrist_angles_R(rotvec_to_mat(_slerp_rotvec(qa[bad], qb[bad], mid)), "right")["twist"]) <= lim[bad]
            hi = np.where(okm, mid, hi)
            lo = np.where(okm, lo, mid)
        frac = np.zeros(len(ks))
        frac[bad] = hi
        brk = np.flatnonzero(np.diff(ks) != 1) + 1
        for seg in np.split(np.arange(len(ks)), brk):
            if len(seg) > 2:
                fs = gaussian_filter1d(maximum_filter1d(frac[seg], 13, mode="nearest"), 2.0, mode="nearest")
                frac[seg] = np.maximum(frac[seg], np.clip(fs, 0.0, 1.0))
        tw = frac > 0
        R21[tw] = rotvec_to_mat(_slerp_rotvec(qa[tw], qb[tw], frac[tw]))
    for j, Rx in ((sh, R17), (el, R19n), (wr, R21)):
        P_new[ks, j] = Ro.from_matrix(Rx).as_rotvec()
    o2 = posed(fit_lead, ks, P_new[ks])
    return P_new, {"palm_moved_m": np.linalg.norm(hc.palm_centre(o2["J"], "right") - hc.palm_centre(o0["J"], "right"), axis=1),
                   "elbow_moved_m": np.linalg.norm(o2["J"][:, el] - E0, axis=1), "wrist_miss_m": np.linalg.norm(o2["J"][:, wr] - Wg, axis=1)}


def collar_fix(fit_lead_in: dict, fit_lead: dict, bound: float = FIX1_COLLAR_BOUND_DEG, knee: float = FIX1_COLLAR_KNEE_DEG) -> tuple[np.ndarray, dict]:
    """the fit's leader poses with the collar bounded against the input (collar_bounded) and the arm re-solved to keep the hand
    (arm_keep_hand) on the frames whose collar changed -> (P (T,55,3), report)"""
    from scipy.spatial.transform import Rotation as Ro
    P_in, P = hc.full_pose(fit_lead_in), hc.full_pose(fit_lead)
    Pb, a0 = collar_bounded(P_in, P, bound, knee)
    ks = np.flatnonzero(np.abs(Pb[:, COLLAR["right"]] - P[:, COLLAR["right"]]).max(1) > 0)
    # the hand kept exactly (no wrist-twist cap: near a straight elbow the re-solved forearm's roll can turn from one frame to the next, and
    # a cap there turned the hand - fingertip snaps of 2-6 m/s at 92.2-92.5 s; without it 12 torso-hold frames twist past 120 deg)
    P2, mv = arm_keep_hand(fit_lead, P, Pb, ks, twist_cap_deg=180.0)
    a1 = np.degrees((Ro.from_rotvec(P_in[:, COLLAR["right"]]).inv() * Ro.from_rotvec(P2[:, COLLAR["right"]])).magnitude())
    rep = {"frames_bounded": int(len(ks)), "collar_deg_max_before_after": [round(float(a0.max()), 2), round(float(a1.max()), 2)],
           "frames_over_bound_before": int((a0 > bound).sum()), "frames_over_15_after": int((a1 > 15.0 + 1e-6).sum()),
           "palm_moved_cm_max": round(float(mv["palm_moved_m"].max()) * 100, 2) if len(ks) else 0.0,
           "palm_moved_gt1cm_frames": int((mv["palm_moved_m"] > 0.01).sum()) if len(ks) else 0,
           "elbow_moved_cm_max": round(float(mv["elbow_moved_m"].max()) * 100, 2) if len(ks) else 0.0}
    return P2, rep


def collar_reclamp(fit_lead: dict, P_in: np.ndarray, P: np.ndarray, bound: float = FIX1_COLLAR_BOUND_DEG,
                   knee: float = FIX1_COLLAR_KNEE_DEG) -> tuple[np.ndarray, dict]:
    """poses P with his right collar's rotation against the input's scaled back to the bound (the same axis) on the frames past it (the
    others untouched), the arm re-solved to keep the hand (arm_keep_hand) -> (P, {"frames", "before_max_deg", "after_max_deg"})"""
    from scipy.spatial.transform import Rotation as Ro
    j = COLLAR["right"]
    a0 = collar_change_deg(P_in, P)
    if a0.max() <= bound + 1e-6:
        return P, {"frames": 0, "before_max_deg": round(float(a0.max()), 3), "after_max_deg": round(float(a0.max()), 3)}
    ks = np.flatnonzero(a0 > bound)
    Ri = Ro.from_rotvec(P_in[ks, j])
    rel = (Ri.inv() * Ro.from_rotvec(P[ks, j])).as_rotvec()
    Pb = np.asarray(P, float).copy()
    Pb[ks, j] = (Ri * Ro.from_rotvec(rel * ((bound - 1e-3) / a0[ks])[:, None])).as_rotvec()
    P2, _ = arm_keep_hand(fit_lead, P, Pb, ks, twist_cap_deg=180.0)
    a1 = collar_change_deg(P_in, P2)
    return P2, {"frames": int(len(ks)), "before_max_deg": round(float(a0.max()), 3), "after_max_deg": round(float(a1.max()), 3)}


def collar_change_deg(P_in: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(T,) his right collar's rotation against the input's (deg)"""
    from scipy.spatial.transform import Rotation as Ro
    return np.degrees((Ro.from_rotvec(P_in[:, COLLAR["right"]]).inv() * Ro.from_rotvec(P[:, COLLAR["right"]])).magnitude())


class SkinLBS:
    """an exported skin (export_unity smplx_<role>_skin.bin: the shape-edited rest mesh, plain LBS - what Unity shows) in the right-handed
    world of the fit: posed with the fit's global rotations / joints (posed())"""

    def __init__(self, role: str, path: Path | None = None, k: int = 4):
        from . import export_unity as eu
        p = Path(path) if path is not None else skin_file(role)
        _, sk = eu.read_skin(p)
        S = np.array([1.0, 1.0, -1.0])
        self.V = np.asarray(sk.vertices, float) * S
        self.Jr = np.asarray(sk.joints, float) * S
        self.faces = np.asarray(sk.faces, np.int64)[:, [0, 2, 1]]
        w = np.asarray(sk.weights, float)
        self.vj = w.argmax(1)
        idx = np.argsort(-w, axis=1)[:, :k]
        ww = np.take_along_axis(w, idx, 1)
        self.idx, self.w = idx, ww / np.maximum(ww.sum(1, keepdims=True), 1e-12)
        self.nv = len(self.V)

    def verts(self, J: np.ndarray, R: np.ndarray, ids: np.ndarray | None = None, batch: int = 32) -> np.ndarray:
        """J (n,55,3), R (n,55,3,3) -> posed vertices (n, nv or len(ids), 3) float32"""
        idx, w, V = (self.idx, self.w, self.V) if ids is None else (self.idx[ids], self.w[ids], self.V[ids])
        n = len(J)
        out = np.zeros((n, len(V), 3), np.float32)
        for a in range(0, n, batch):
            Jb, Rb = J[a:a + batch], R[a:a + batch]
            acc = np.zeros((len(Jb), len(V), 3))
            for c in range(idx.shape[1]):
                j = idx[:, c]
                acc += w[None, :, c, None] * (np.einsum("nvab,vb->nva", Rb[:, j], V - self.Jr[j]) + Jb[:, j])
            out[a:a + batch] = acc
        return out


def skin_file(role: str) -> Path:
    """the exported skin of a dancer (copied once from the v43 side export into v4/contacts/skins_v43/: a re-export rewrites the side export)"""
    import shutil
    p = SKINS_V43 / f"smplx_{role}_skin.bin"
    if not p.exists():
        src = WORK / "export" / SIDE_CAPTURE_V43 / f"smplx_{role}_skin.bin"
        SKINS_V43.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, p)
    return p


# her parts by the dominant joint of the exported skin's vertices: torso + collars (where his palm rests; the ray targets), + upper arms (the
# penetration of his arm), her neck / head (his hand never in it)
FIX1_HER_TORSO = (PELVIS, SPINE1, SPINE2, SPINE3, COLLAR["left"], COLLAR["right"])
FIX1_HER_ARMS = (SHOULDER["left"], SHOULDER["right"])
FIX1_HER_NECK = (NECK, HEAD, 22, 23, 24)
FIX1_LEAD_HAND = (WRIST["right"], *range(40, 55))


@dataclass
class Fix1Cfg:
    lift_m: float = 0.012                # the palm centre this far off her exported skin (the palm's half thickness)
    thetas_deg: tuple = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)   # the grid of targets: from her left side (0) round to her back (90) ...
    dys_m: tuple = (-0.04, 0.0, 0.04, 0.08, 0.12)                     # ... at these heights above spine 3 along her spine (-4 cm: her flank
                                         #   below the armpit, the leans' fallback where the arm cannot reach round her back)
    ulb_ideal: tuple = (50.0, 0.09)      # (theta deg, dy m) the class's best place: her left shoulder blade (the ULB sites sit at 36-64 deg, 7-10 cm) ...
    cb_ideal: tuple = (30.0, 0.06)       # ... in the counterbalance leans her left back edge at her armpit (the back of her shoulder,
                                         #   shoulder_back_L at 36 deg / 9.6 cm, is past his straight arm or under her raised arm there)
    back_ideal: tuple = (90.0, 0.04)     # ... wrap / behind: her upper back
    loc_scale: tuple = (25.0, 0.05)      # the location cost = ((theta - ideal) / 25)^2 + ((dy - ideal) / 0.05)^2 ...
    max_dy_m: float = 0.11               # ... + ((dy - this) / 0.02)^2 above it (not over the top of her shoulder)
    collar_deg: float = 14.0             # the IK's collar budget against the INPUT collar (<= FIX1_COLLAR_BOUND_DEG)
    reach_tol_m: float = 0.01            # a target the IK misses by more costs 200 / m beyond this (+ 10 beyond 2 cm)
    straight_ratio: float = 0.99         # reach ratios past this cost (ratio - this) x 100
    switch_cost: float = 2.0             # the target Viterbi: a change of candidate costs this + 8 x the jump (m, her frame)
    sigma_frames: float = 4.0            # the target track (and the swivel) smoothed in her frame ...
    rate_mps: float = 0.30               # ... and its speed against her limited to this
    swivels_deg: tuple = (-120.0, -90.0, -60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0, 90.0, 120.0)
    w_pen: float = 400.0                 # the swivel cost per cm of his forearm / upper arm inside her (beyond the allowances) ...
    upper_arm_allow_m: float = 0.015     # ... her upper arms give way this much to his forearm (her arm rests on his: PLAN H4 allowance)
    w_coco_elbow: float = 60.0           # ... per m from the triangulated COCO elbow
    w_coco_wrist: float = 12.0           # the target cost per m of the IK wrist from the triangulated COCO wrist (<= 30 cm; counterbalance leans)
    w_target_pen: float = 2.0            # the target cost per cm^2 of his hand (+ half per cm^2 of his forearm beyond the allowance) in her
    w_sag: float = 60.0                  # ... per m of elbow below the shoulder-wrist line beyond sag_free_m (counterbalance leans: level arm)
    sag_free_m: float = 0.03
    w_swivel: float = 0.4                # ... per rad away from the hint's side
    w_twist_sw: float = 0.05             # ... per deg of wrist twist beyond 45 (+ 0.6 x the elbow's roll beyond 45)
    v_ramp_mps: float = 0.22             # the ramps: the move at the core's ends at this mean speed (a raised cosine peaks at 1.57 x)
    blend_mps: float = 0.25              # the palm's way between the base and the solved arm (blends inside a window, the segment selection)
    select: bool = False                 # keep the solved window only on its stretches better than the base (fix1_select; off: blends of two
                                         #   unlike arms pass the hand through her)
    ramp_min: int = 15
    ramp_max: int = 90                   # (3 s: a move from her front to her back turns the hand over on the ramp)
    move_full_m: float = 0.04            # the solved arm blends in fully where the target moved this far from the base palm
    pad_s: float = 0.3
    merge_s: float = 1.0
    min_s: float = 0.4
    sag_bad_m: float = 0.08              # a counterbalance frame whose elbow sags more needs the re-pose
    max_core_s: float = 8.0              # long runs are solved as consecutive chunks of at most this
    push_rounds: int = 4                 # the hand pushed out of her (fix1_window, last step): rounds ...
    push_tol_m: float = 0.012            # ... on the frames where his hand's skin is deeper in her than this ...
    push_margin_m: float = 0.006         # ... to this far outside her (along the deepest point's way out) ...
    push_max_m: float = 0.08             # ... never more than this in all
    device: str = "cuda"


def spine_frame(J: np.ndarray) -> np.ndarray:
    """(n,3,3) columns = the dancer's left (across the shoulders, orthogonal to the spine), the spine (spine 3 -> neck: her up when she leans),
    front - the frame of the target grid (torso_frame keeps the world's up)"""
    sp = _unit(J[:, NECK] - J[:, SPINE3])
    x = J[:, SHOULDER["left"]] - J[:, SHOULDER["right"]]
    x = _unit(x - (x * sp).sum(1, keepdims=True) * sp)
    z = np.cross(x, sp)
    return np.stack([x, sp, z], 2)


# the configurations a window tries in turn (over Fix1Cfg): as is; her upper arms giving only 0.5 cm and the leans' hand lower on her flank
# (his forearm under her raised arm); that and the COCO elbow / wrist leading harder
FIX1_VARIANTS = ({}, {"upper_arm_allow_m": 0.005, "cb_ideal": (25.0, 0.02), "w_target_pen": 3.0},
                 {"upper_arm_allow_m": 0.005, "cb_ideal": (25.0, 0.02), "w_target_pen": 3.0, "w_coco_elbow": 150.0, "w_coco_wrist": 30.0})
FIX1_SHOULDER_DIRS = {"sh_b": (0.0, 0.0, -1.0), "sh_bu": (0.0, 0.6, -1.0), "sh_bd": (0.0, -0.5, -1.0)}
# rays from her left shoulder joint (her spine frame): the back of her left shoulder (not its lateral side: his forearm would cross her arm)


def fix1_grid(Jf: np.ndarray, VF: np.ndarray, faces_torso: np.ndarray, cfg: "Fix1Cfg", device: str = "cpu",
              faces_shoulder: np.ndarray | None = None) -> dict:
    """the target grid on her (exported) skin: per frame and (theta, dy) the exit of the ray from her spine axis (spine 3 + dy along the spine)
    toward her left (theta 0) round to her back (90) through her torso + collars (faces_torso), lifted lift_m along the surface normal; and
    (faces_shoulder: her torso + collars + left upper arm) the exits of the rays from her left shoulder joint toward the back of her left
    shoulder (FIX1_SHOULDER_DIRS; the leans' connection: counterbalance.json's lead_r_hand-follow_l_shoulder)
    -> {"X" (W,G,3), "N" (W,G,3) outward normals, "ok" (W,G), "theta", "dy" (G,) the grid's, "theta_f", "dy_f" (W,G) every candidate's
    place in her spine frame, "names" (G,)}"""
    W = len(Jf)
    Fr = spine_frame(Jf)
    th, dy = np.meshgrid(np.radians(cfg.thetas_deg), np.asarray(cfg.dys_m, float), indexing="ij")
    th, dy = th.ravel(), dy.ravel()
    G = len(th)
    X = np.zeros((W, G, 3))
    N = np.zeros((W, G, 3))
    ok = np.zeros((W, G), bool)
    for g in range(G):
        O = Jf[:, SPINE3] + dy[g] * Fr[:, :, 1]
        D = np.cos(th[g]) * Fr[:, :, 0] - np.sin(th[g]) * Fr[:, :, 2]
        t, fi = hc.ray_first_hit(O, D, VF, faces_torso, device)
        hit = np.isfinite(t) & (fi >= 0) & (t < 0.40)
        if not hit.any():
            continue
        n = face_normals_np(np.asarray(VF[hit], float), faces_torso, fi[hit])
        n = np.where(((n * D[hit]).sum(1) < 0)[:, None], -n, n)
        X[hit, g] = O[hit] + t[hit, None] * D[hit] + cfg.lift_m * n
        N[hit, g] = n
        ok[hit, g] = True
    names = [f"t{int(round(np.degrees(a)))}_y{int(round(b * 100))}" for a, b in zip(th, dy)]
    if faces_shoulder is not None:
        Xs, Ns, oks = [], [], []
        for nm, dv in FIX1_SHOULDER_DIRS.items():
            dv = _unit(np.asarray(dv, float)[None])[0]
            O = Jf[:, SHOULDER["left"]]
            D = np.einsum("kij,j->ki", Fr, dv)
            t, fi = hc.ray_first_hit(O, D, VF, faces_shoulder, device)
            hit = np.isfinite(t) & (fi >= 0) & (t < 0.25)
            x_, n_ = np.zeros((W, 3)), np.zeros((W, 3))
            if hit.any():
                n = face_normals_np(np.asarray(VF[hit], float), faces_shoulder, fi[hit])
                n = np.where(((n * D[hit]).sum(1) < 0)[:, None], -n, n)
                x_[hit] = O[hit] + t[hit, None] * D[hit] + cfg.lift_m * n
                n_[hit] = n
            Xs.append(x_)
            Ns.append(n_)
            oks.append(hit)
            names.append(nm)
        X = np.concatenate([X, np.stack(Xs, 1)], 1)
        N = np.concatenate([N, np.stack(Ns, 1)], 1)
        ok = np.concatenate([ok, np.stack(oks, 1)], 1)
    loc = np.einsum("kji,kgj->kgi", Fr, X - Jf[:, SPINE3][:, None])
    theta_f = np.degrees(np.arctan2(-loc[..., 2], loc[..., 0]))
    dy_f = loc[..., 1]
    return {"X": X, "N": N, "ok": ok, "theta": np.degrees(th), "dy": dy, "theta_f": theta_f, "dy_f": dy_f, "names": names}


def fix1_loc_cost(theta: np.ndarray, dy: np.ndarray, cls: np.ndarray, cfg: "Fix1Cfg") -> np.ndarray:
    """(W,G) the location cost of the targets (theta / dy: (G,) or per frame (W,G), her spine frame) for each frame's class (0 at the
    class's ideal place)"""
    W = len(cls)
    theta = np.broadcast_to(np.asarray(theta, float), (W,) + np.shape(theta)[-1:])
    dy = np.broadcast_to(np.asarray(dy, float), (W,) + np.shape(dy)[-1:])
    ideal = {"ulb": cfg.ulb_ideal, "cb": cfg.cb_ideal, "back": cfg.back_ideal}
    out = np.full(theta.shape, 50.0)
    for c, (t0, d0) in ideal.items():
        m = np.asarray(cls) == c
        if m.any():
            out[m] = ((theta[m] - t0) / cfg.loc_scale[0]) ** 2 + ((dy[m] - d0) / cfg.loc_scale[1]) ** 2
            # never over the top of her shoulder (the arm hooking over it): above max_dy_m costs steeply
            out[m] += (np.maximum(0.0, dy[m] - cfg.max_dy_m) / 0.02) ** 2
    return out


def elbow_sag(S: np.ndarray, E: np.ndarray, W_: np.ndarray) -> np.ndarray:
    """(n,) how far the elbow hangs below the shoulder-wrist line (m, world up; < 0: above it)"""
    u = _unit(W_ - S)
    c = S + ((E - S) * u).sum(1, keepdims=True) * u
    return -(E - c)[:, 1]


def fix1_ik(J: np.ndarray, R: np.ndarray, R_in14: np.ndarray, rest: dict, X: np.ndarray, n_out: np.ndarray, fdir: np.ndarray,
            ehint: np.ndarray, cfg: "Fix1Cfg", hinge_in: np.ndarray | None = None) -> dict:
    """arm_ik onto palm targets X (W,3) facing her (n_out: her outward normal) with the collar budget counted against the INPUT's collar
    (R_in14: the input collar's local rotation (W,3,3); the collar starts there and turns at most cfg.collar_deg)"""
    G9 = R[:, SPINE3]
    G14_in = G9 @ R_in14
    return arm_ik(J[:, COLLAR["right"]], G9, G14_in, rest, X, n_out, fdir, ehint, hinge_in, collar_max_deg=cfg.collar_deg,
                  follow_forearm=False, side_from_input=None, bend_soft=True)


def fix1_pose(P: np.ndarray, ik: dict) -> np.ndarray:
    """P (W,55,3) with his right collar / shoulder / elbow / wrist from fix1_ik (the collar ALWAYS: it starts at the input's)"""
    from scipy.spatial.transform import Rotation as Ro
    P = np.asarray(P, float).copy()
    for j, k in ((COLLAR["right"], "R14"), (SHOULDER["right"], "R17"), (ELBOW["right"], "R19"), (WRIST["right"], "R21")):
        P[:, j] = Ro.from_matrix(ik[k]).as_rotvec()
    return P


def her_surfaces(skinF: "SkinLBS", device: str) -> dict:
    """her exported skin's GpuSurfaces: "torso_arms" (torso + collars + upper arms: his arm's penetration), "neck" (her neck / head), and the
    torso + collar faces (the ray targets)"""
    vj = skinF.vj
    out = {}
    for nm, js in (("torso_arms", FIX1_HER_TORSO + FIX1_HER_ARMS), ("neck", FIX1_HER_NECK), ("torso", FIX1_HER_TORSO)):
        out[nm] = hc.GpuSurface(skinF.faces, skinF.nv, device, vmask=np.isin(vj, js))
    out["lua"] = hc.GpuSurface(skinF.faces, skinF.nv, device, vmask=vj == SHOULDER["left"])
    out["all"] = hc.GpuSurface(skinF.faces, skinF.nv, device)
    out["faces_torso"] = skinF.faces[np.isin(vj, FIX1_HER_TORSO)[skinF.faces].all(1)]
    lsh = FIX1_HER_TORSO + (SHOULDER["left"],)
    out["faces_shoulder"] = skinF.faces[np.isin(vj, lsh)[skinF.faces].all(1)]
    out["torso_lsh"] = hc.GpuSurface(skinF.faces, skinF.nv, device, vmask=np.isin(vj, lsh))
    out["vj"] = vj
    return out


def surface_depth(surf, Pts: np.ndarray, VF: np.ndarray, allow: np.ndarray | None = None, vallow: np.ndarray | None = None,
                  exact: bool = True, bs: int = 8) -> np.ndarray:
    """(W,n) how deep each point of Pts (W,n,3) is inside her surface (m, >= 0; the winding-number inside test, within 15 cm of her surface
    when exact, within 9 cm otherwise - the interpolated normal's side alone reads points far off her masked open surface as deep inside),
    minus a per-point allowance (allow (n,)) or the closest vertex's allowance (vallow (nv,))"""
    import torch
    dev = surf.dev
    Pt, Vt = torch.as_tensor(np.asarray(Pts, np.float32), device=dev), torch.as_tensor(np.asarray(VF, np.float32), device=dev)
    sds, verts = [], []
    with torch.no_grad():
        for a in range(0, Pt.shape[0], bs):
            q = surf.query(Pt[a:a + bs], Vt[a:a + bs], wn_within=0.03, far=0.15 if exact else 0.09, inside_test=True)
            sds.append(q["sd"])
            verts.append(q["vert"])
    q = {"sd": torch.cat(sds, 0), "vert": torch.cat(verts, 0)}
    d = (-q["sd"]).cpu().numpy().astype(float)
    if allow is not None:
        d = d - np.asarray(allow, float)[None]
    if vallow is not None:
        d = d - np.asarray(vallow, float)[q["vert"].cpu().numpy()]
    return np.maximum(d, 0.0)


def hand_way_out(surfs: dict, Pts: np.ndarray, VF: np.ndarray, names: tuple = ("torso_arms", "neck"), bs: int = 8,
                 axes: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """per frame the deepest of the points Pts (W,n,3) inside her masked surfaces (winding numbers) and its way out: (W,) depth (m, 0 when
    none is inside), (W,3) the unit vector from that point to its closest point on her surface (zeros when none) - or, for a surface named
    in axes ({name: (A, B)} (W,3) each: her neck -> head), away from that segment (her neck patch is an open tube: its closest point may be
    the tube's lower rim, whose way leads down into her shoulders)"""
    import torch
    W = len(Pts)
    axes = axes or {}
    best_d, best_u = np.zeros(W), np.zeros((W, 3))
    for nm in names:
        s = surfs[nm]
        q = hc.query_batched(s, torch.as_tensor(np.asarray(Pts, np.float32), device=s.dev),
                             torch.as_tensor(np.asarray(VF, np.float32), device=s.dev), bs=bs, wn_within=0.03, inside_test=True)
        sd = q["sd"].cpu().numpy().astype(float)
        qq = q["q"].cpu().numpy().astype(float)
        d = np.maximum(-sd, 0.0)
        j = d.argmax(1)
        dj = d[np.arange(W), j]
        pj = np.asarray(Pts, float)[np.arange(W), j]
        uj = _unit(qq[np.arange(W), j] - pj)
        if nm in axes:
            A_, B_ = (np.asarray(x, float) for x in axes[nm])
            ab = B_ - A_
            tt = np.clip(((pj - A_) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
            uj = _unit(pj - (A_ + tt[:, None] * ab), uj)
        bt = dj > best_d
        best_d = np.where(bt, dj, best_d)
        best_u = np.where(bt[:, None], uj, best_u)
    return best_d, best_u


def viterbi_tr(cost: np.ndarray, allowed: np.ndarray, tr: np.ndarray) -> np.ndarray:
    """per frame the column of least summed cost among the allowed ones with a transition cost matrix tr (C,C)"""
    score = np.where(allowed, -np.asarray(cost, float), -1e6)
    return viterbi(score, np.asarray(tr, float))


def fix1_project(surf, X: np.ndarray, VF: np.ndarray, lift_m: float, Jf: np.ndarray, sigma: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """a palm track X (W,3) laid onto her surface: the closest point + lift_m along her smooth normal there, the normals smoothed in her frame
    (sigma frames) -> (X, normals)"""
    import torch
    from scipy.ndimage import gaussian_filter1d
    q = hc.query_batched(surf, torch.as_tensor(np.asarray(X, np.float32)[:, None], device=surf.dev),
                         torch.as_tensor(np.asarray(VF, np.float32), device=surf.dev), bs=16, inside_test=False)
    qq = q["q"][:, 0].cpu().numpy().astype(float)
    nq = _unit(q["nq"][:, 0].cpu().numpy().astype(float))
    Fr = torso_frame(Jf)
    if len(X) > 3 and sigma > 0:
        nl = gaussian_filter1d(np.einsum("kji,kj->ki", Fr, nq), sigma, axis=0, mode="nearest")
        nq = _unit(np.einsum("kij,kj->ki", Fr, nl))
    return qq + lift_m * nq, nq


def lead_arm_ids(skinL: "SkinLBS") -> dict:
    """his exported skin's vertices of the right hand, forearm and upper arm (dominant joint)"""
    import trimesh
    nrm = trimesh.Trimesh(skinL.V, skinL.faces, process=False).vertex_normals
    return {"hand": np.flatnonzero(np.isin(skinL.vj, FIX1_LEAD_HAND)), "fore": np.flatnonzero(skinL.vj == ELBOW["right"]),
            "upper": np.flatnonzero(skinL.vj == SHOULDER["right"]),
            # the palm heel: his wrist-dominated vertices on the palm side (the rest pose has the palms down)
            "palm": np.flatnonzero((skinL.vj == WRIST["right"]) & (nrm[:, 1] < -0.5))}


def skin_contact(fit_lead: dict, P: np.ndarray, ks: np.ndarray, kf: dict, VF: np.ndarray, skinL: "SkinLBS", ids: dict, surfs: dict) -> dict:
    """his right hand / forearm (his exported skin, the poses P on ks) against her exported skin VF (W,nv,3), winding-number inside tests,
    the independent check's measures -> per frame (m): "hand" depth into her torso + collars + upper arms, "fore" his forearm's depth into
    them, "fore_lua" his forearm in her left upper arm, "neck" his hand in her neck / head, "palm_gap" the palm heel's (his wrist-dominated
    palm-side vertices) signed distance to her (> 0 floating)"""
    o = posed(fit_lead, ks, P)
    allv = np.concatenate([ids["hand"], ids["fore"]])
    VL = skinL.verts(o["J"], o["R"], allv)
    nh = len(ids["hand"])
    dh = surface_depth(surfs["torso_arms"], VL[:, :nh], VF)
    df = surface_depth(surfs["torso_arms"], VL[:, nh:], VF)
    dl = surface_depth(surfs["lua"], VL[:, nh:], VF)
    dn = surface_depth(surfs["neck"], VL[:, :nh], VF)
    import torch
    pm = np.isin(ids["hand"], ids["palm"])
    q = hc.query_batched(surfs["all"], torch.as_tensor(VL[:, :nh][:, pm], device=surfs["all"].dev),
                         torch.as_tensor(np.asarray(VF, np.float32), device=surfs["all"].dev), bs=8, wn_within=0.03, inside_test=True)
    gap = q["sd"].cpu().numpy().astype(float).min(1)
    return {"hand": dh.max(1), "fore": df.max(1), "fore_lua": dl.max(1), "neck": dn.max(1), "palm_gap": gap}


def fix1_metrics(fit_lead: dict, P: np.ndarray, ks: np.ndarray, kf: dict, sites_f: dict, cls: np.ndarray) -> dict:
    """the verifier's location measures of the palm per frame: z in her torso frame (m, + = her front), the distance to her upper-left back
    (ULB5 sites, raw), on ULB (<= 8 cm and z <= +4 cm; the counterbalance class also the top of her left shoulder), the elbow sag"""
    o = posed(fit_lead, ks, P)
    pc = hc.palm_centre(o["J"], "right")
    z = in_frame(pc, kf["J"])[:, 2]
    pts = {s: site_world(sites_f[s], kf["J"], kf["R"])[0] for s in ULB5 + ("shoulder_top_L",)}
    dU = np.min([np.linalg.norm(pts[s] - pc, axis=1) for s in ULB5], 0)
    dT = np.linalg.norm(pts["shoulder_top_L"] - pc, axis=1)
    ulb = (dU <= 0.08) & (z <= 0.04)
    ulb_cb = ulb | ((np.asarray(cls) == "cb") & (dT <= 0.08) & (z <= 0.04))
    return {"z": z, "d_ulb": dU, "ulb": ulb, "ulb_cb": ulb_cb, "front": z > 0.04, "back": ulb_cb | (z < -0.06),
            "sag": elbow_sag(o["J"][:, SHOULDER["right"]], o["J"][:, ELBOW["right"]], o["J"][:, WRIST["right"]]), "J": o["J"], "pc": pc}


def coco_elbow_track(fit_in: dict, cache: Path | None = None, which: str = "elbow") -> np.ndarray:
    """(T,3) the triangulated, bias-corrected COCO right elbow (or wrist) of the leader where >= 2 phones see it (NaN elsewhere), over the
    whole film (cached in v4/contacts/coco_<which>_lead_r.npz)"""
    p = cache or (OUT / f"coco_{which}_lead_r.npz")
    if p.exists():
        with np.load(p) as z:
            return z["X"].copy()
    obs = hc.Obs()
    judge = hc.CocoJudge()
    bias = hc.calibrate_coco_bias(fit_in, obs, judge)
    T = len(fit_in["lead"]["t"])
    X = np.full((T, 3), np.nan)
    for a in range(0, T, 1500):
        ks = np.arange(a, min(T, a + 1500))
        ct = hc.coco_targets(fit_in["lead"], "lead", "right", ks, judge, obs, list(obs.cams), bias)
        Xe, ve, nve = ct[which]
        X[ks] = np.where((ve & (nve >= 2))[:, None], Xe, np.nan)
    np.savez_compressed(p, X=X)
    return X


def fix1_context(base: str = "v43", device: str = "cuda", collar: bool = True) -> dict:
    """everything the fix-1 windows share: the input fit and poses, the base poses (the base fit with the collar bounded against the input,
    collar_fix), the holds / classes, her posed joints, her and his exported skins, her surfaces, the sites, his arm's rest, the COCO elbow"""
    fit_in = load_fit("input")
    fb = load_fit(base)
    T = len(fit_in["lead"]["t"])
    P_in = hc.full_pose(fit_in["lead"])
    if collar:
        P_base, crep = collar_fix(fit_in["lead"], fb["lead"])
    else:
        P_base, crep = hc.full_pose(fb["lead"]), {}
    H = holds_with_overrides(load_holds(), V43_HOLD_OVERRIDES)
    cls = v43_classes(H["names"], np.arange(T), counterbalance_intervals())
    skinF, skinL = SkinLBS("follow"), SkinLBS("lead")
    return {"fit_in": fit_in, "fit_lead": fb["lead"], "P_in": P_in, "P_base": P_base, "collar_report": crep, "names": np.asarray(H["names"]),
            "cls": cls, "kf": posed(fit_in["follow"]), "skinF": skinF, "skinL": skinL, "ids": lead_arm_ids(skinL),
            "surfs": her_surfaces(skinF, device), "sites_f": site_bodies(fit_in)["follow"], "rest": arm_rest(fb["lead"]["betas"]),
            "elbow_obs": coco_elbow_track(fit_in), "wrist_obs": coco_elbow_track(fit_in, which="wrist"), "device": device, "base": base}


def fix1_window(ctx: dict, win: dict, cfg: "Fix1Cfg") -> dict:
    """one fix-1 window (see the section): ctx = fix1_context(); win = {"core": [c0, c1], "span": [s0, s1]} frames
    -> {"ks", "P" (the window's new poses), "target", "sel", "phi", "stats"}"""
    from scipy.ndimage import gaussian_filter1d
    fit_lead = ctx["fit_lead"]
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    ks = np.arange(s0, s1 + 1)
    W = len(ks)
    wr = envelope(c0 - s0, c1 - c0 + 1, s1 - c1)
    core = wr > 0.999
    ci = np.flatnonzero(core)
    Pb = ctx["P_base"][ks].copy()
    Pi = ctx["P_in"][ks]
    cls = ctx["cls"][ks]
    o = posed(fit_lead, ks, Pb)
    J, R = o["J"], o["R"]
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    Jf = kf["J"]
    VF = ctx["skinF"].verts(Jf, kf["R"])
    surfs = ctx["surfs"]
    Fr = torso_frame(Jf)
    xh, yu, zh = Fr[:, :, 0], Fr[:, :, 1], Fr[:, :, 2]
    rest = ctx["rest"]
    R_in14 = rotvec_to_mat(Pi[:, COLLAR["right"]])
    pc0 = hc.palm_centre(J, "right")
    m0 = fix1_metrics(fit_lead, Pb, ks, kf, ctx["sites_f"], cls)
    # ---- the base frames that may stay: on her upper-left back / back, not in front of her, the counterbalance elbow level
    ok = (cls != "") & (m0["z"] <= 0.0) & np.where(cls == "back", m0["z"] <= 0.0, m0["ulb_cb"]) & ~((cls == "cb") & (m0["sag"] > cfg.sag_bad_m))
    if "bad" in win:
        ok &= ~np.isin(ks, np.asarray(win["bad"], int))
    # ... and not in her (his hand > 1.5 cm in her, his forearm > 2 cm in her left upper arm, his hand in her neck)
    sc0 = skin_contact(fit_lead, Pb, ks, kf, VF, ctx["skinL"], ctx["ids"], surfs)
    ok &= (sc0["hand"] <= 0.015) & (sc0["fore_lua"] <= 0.02) & (sc0["neck"] <= 0.005)
    # ---- the grid on her exported skin
    grid = fix1_grid(Jf, VF, surfs["faces_torso"], cfg, ctx["device"], faces_shoulder=surfs["faces_shoulder"])
    G = grid["X"].shape[1]
    loc = fix1_loc_cost(grid["theta_f"], grid["dy_f"], cls, cfg)
    face = couple_geometry(J, Jf)["facing"]
    ftf = face < HoldCfg().face_to_face
    fdir_ftf = _unit(-0.6 * xh - 0.8 * zh + 0.25 * yu)
    S_b = J[:, SHOULDER["right"]]

    def fdir_for(X, n):
        fa = _unit(X - S_b)
        fa = _unit(fa - (fa * n).sum(1, keepdims=True) * n + 0.3 * yu)
        return np.where(ftf[:, None], fdir_ftf, fa)
    # the elbow hint: the triangulated COCO elbow where it is seen (>= 2 phones, within 35 cm of the base's), else (face to face, the closed
    # hold) her left flank / (the counterbalance leans) a level arm bending outward / (else) the base elbow
    hint_flank = np.einsum("kij,j->ki", Fr, np.array([0.24, -0.10, 0.06])) + Jf[:, SPINE3]
    hint_cb = 0.5 * (S_b + pc0) + 0.12 * xh + 0.04 * yu         # a level arm bending outward (her left = his right, face to face)
    hint = np.where((cls == "cb")[:, None], hint_cb, np.where(ftf[:, None], hint_flank, J[:, ELBOW["right"]]))
    Eo = ctx["elbow_obs"][ks] if ctx.get("elbow_obs") is not None else np.full((W, 3), np.nan)
    okE = np.isfinite(Eo).all(1) & (np.linalg.norm(np.nan_to_num(Eo) - J[:, ELBOW["right"]], axis=1) < 0.35)
    if okE.any():
        wo = np.clip(gaussian_filter1d(okE.astype(float), 3.0, mode="nearest"), 0, 1) if W > 3 else okE.astype(float)
        idx = np.flatnonzero(okE)
        Ef = np.stack([np.interp(np.arange(W), idx, Eo[idx, d]) for d in range(3)], 1)
        hint = wo[:, None] * Ef + (1 - wo[:, None]) * hint
    hint = gaussian_filter1d(hint, 2.0, axis=0, mode="nearest") if W > 3 else hint
    Wo = ctx["wrist_obs"][ks] if ctx.get("wrist_obs") is not None else np.full((W, 3), np.nan)
    okW = np.isfinite(Wo).all(1) & (np.linalg.norm(np.nan_to_num(Wo) - Jf[:, SPINE3], axis=1) < 0.45)
    # the INPUT's palm where the input is on her upper-left back (2:30: the detected hand was there, only too deep): laid onto her skin,
    # a candidate without a location cost (the verifier's F5: restore the input where it was on the upper-left back)
    oi_ = posed(fit_lead, ks, Pi)
    mi_ = fix1_metrics(fit_lead, Pi, ks, kf, ctx["sites_f"], cls)
    xi_, ni_ = fix1_project(surfs["torso_lsh"], hc.palm_centre(oi_["J"], "right"), VF, cfg.lift_m, Jf)
    inp_ok = mi_["ulb_cb"] & (mi_["z"] <= 0.02) & (cls != "")
    grid["X"] = np.concatenate([grid["X"], xi_[:, None]], 1)
    grid["N"] = np.concatenate([grid["N"], ni_[:, None]], 1)
    grid["ok"] = np.concatenate([grid["ok"], inp_ok[:, None]], 1)
    grid["names"] = list(grid["names"]) + ["input"]
    loc = np.concatenate([loc, np.full((W, 1), -0.3)], 1)
    G = grid["X"].shape[1]
    # ---- target candidates: keep (the base palm, where ok) + the grid
    C = 1 + G
    Xc = np.concatenate([pc0[:, None], grid["X"]], 1)
    Nc = np.concatenate([_unit(-hc.palm_normal(J, "right"))[:, None], grid["N"]], 1)
    cost = np.zeros((W, C))
    allowed = np.zeros((W, C), bool)
    allowed[:, 0] = ok | ~core
    cost[:, 1:] = loc + 0.5
    allowed[:, 1:] = grid["ok"] & (cls != "")[:, None]
    reach = np.full((W, C), np.nan)
    bendc = np.zeros((W, C))
    # fingers: the open template (a flat hand, fingers together: on her back, on the back of her shoulder in the leans - never curled into her)
    flex_o, fixed_o = hc.finger_template(fit_lead["betas"], "right", "open")
    fing = np.tile((flex_o + fixed_o)[None], (W, 1, 1))
    ids = ctx["ids"]
    arm_ids = np.concatenate([ids["fore"], ids["upper"][::2]])
    hand_sub = ids["hand"][::3]
    vall_arm = np.where(np.isin(surfs["vj"], FIX1_HER_ARMS), cfg.upper_arm_allow_m, 0.0)
    vall_h = np.where(np.isin(surfs["vj"], FIX1_HER_ARMS), 0.005, 0.0)
    for g in range(1, C):
        m = allowed[:, g]
        if not m.any():
            continue
        ik = fix1_ik(J[m], R[m], R_in14[m], rest, Xc[m, g], Nc[m, g], fdir_for(Xc[:, g], Nc[:, g])[m], hint[m], cfg)
        miss = ik["residual_m"]
        ratio = np.linalg.norm(ik["W"] - ik["S"], axis=1) / (np.linalg.norm(rest["r19"]) + np.linalg.norm(rest["r21"]))
        reach[m, g] = ratio
        bendc[m, g] = ik["bend_deg"]
        cost[m, g] += 200.0 * np.maximum(0.0, miss - cfg.reach_tol_m) + np.where(miss > 0.02, 10.0, 0.0) \
            + 100.0 * np.maximum(0.0, ratio - cfg.straight_ratio) \
            + np.maximum(0.0, ik["bend_deg"] - 70.0) / 10.0
        # the counterbalance leans: the phones see his straight arm in profile - the triangulated COCO wrist leads the height / depth
        cw = m & okW & (cls == "cb")
        if cw.any():
            dw = np.linalg.norm(ik["Wt"] - np.nan_to_num(Wo[m]), axis=1)
            cost[cw, g] += cfg.w_coco_wrist * np.minimum(dw[cw[m]], 0.30)
        # his hand / forearm at this target (the hint's swivel) in her (her exported skin; her upper arms give way a little): a target
        # under her raised upper arm (the leans: her arm rests on his) costs what it sinks in
        mm = np.flatnonzero(m & core)
        if len(mm) and cfg.w_target_pen > 0:
            sub = mm[::3] if len(mm) > 40 else mm
            Pg = fix1_pose(Pb[sub], {k_: v_[np.searchsorted(np.flatnonzero(m), sub)] for k_, v_ in ik.items()
                                     if k_ in ("R14", "R17", "R19", "R21")})
            Pg[:, 40:55] = fing[sub]
            og = posed(fit_lead, ks[sub], Pg)
            VHg_ = ctx["skinL"].verts(og["J"], og["R"], hand_sub)
            dh_ = surface_depth(surfs["torso_arms"], VHg_, VF[sub], exact=False, vallow=vall_h).max(1)
            dh_ = np.maximum(dh_, 3.0 * surface_depth(surfs["neck"], VHg_, VF[sub], exact=False).max(1))     # never into her neck
            da_ = surface_depth(surfs["torso_arms"], ctx["skinL"].verts(og["J"], og["R"], ids["fore"][::2]), VF[sub], exact=False,
                                vallow=vall_arm).max(1)
            pen_c = cfg.w_target_pen * ((dh_ * 100) ** 2 + 0.5 * (da_ * 100) ** 2)
            cost[mm, g] += np.interp(mm, sub, pen_c)
    # the Viterbi over the core (the ramps keep: their change is the core edge's faded)
    xl = np.einsum("kji,kcj->kci", Fr, Xc - Jf[:, SPINE3][:, None])
    mean_l = np.array([xl[core, c].mean(0) for c in range(C)])
    tr = cfg.switch_cost + 8.0 * np.linalg.norm(mean_l[:, None] - mean_l[None], axis=-1)
    np.fill_diagonal(tr, 0.0)
    sel = np.zeros(W, int)
    sel[ci] = viterbi_tr(cost[ci], allowed[ci], tr)
    ar = np.arange(W)
    # ---- the track: the change against the base palm in her frame, smoothed and rate-limited over the core, faded on the ramps
    Xs = Xc[ar, sel]
    Xl = np.einsum("kji,kj->ki", Fr, Xs - Jf[:, SPINE3])
    loc0 = np.einsum("kji,kj->ki", Fr, pc0 - Jf[:, SPINE3])
    xc = Xl[ci]
    if len(ci) > 3:
        xc = gaussian_filter1d(xc, cfg.sigma_frames, axis=0, mode="nearest")
    xc = speed_limit_track(xc, cfg.rate_mps / HZ)
    dc = xc - loc0[ci]
    dl = np.zeros((W, 3))
    dl[ci] = dc
    # the ramps: the core edge's change faded by the ramp weight in her CYLINDRICAL coordinates (angle round her vertical axis, height,
    # radius): a palm going from her front to her back goes round her side, never through her

    def cyl(x):
        return np.stack([np.arctan2(-x[..., 2], x[..., 0]), x[..., 1], np.hypot(x[..., 0], x[..., 2])], -1)

    def cart(c):
        return np.stack([c[..., 2] * np.cos(c[..., 0]), c[..., 1], -c[..., 2] * np.sin(c[..., 0])], -1)
    for idx, e in ((np.arange(0, ci[0]), ci[0]), (np.arange(ci[-1] + 1, W), ci[-1])):
        if not len(idx):
            continue
        c0_, c1_ = cyl(loc0[e]), cyl(xc[0] if e == ci[0] else xc[-1])
        dcy = c1_ - c0_
        dcy[0] = (dcy[0] + np.pi) % (2 * np.pi) - np.pi
        cb_ = cyl(loc0[idx]) + wr[idx, None] * dcy[None]
        dl[idx] = cart(cb_) - loc0[idx]
    tgt = pc0 + np.einsum("kij,kj->ki", Fr, dl)
    moved = np.linalg.norm(dl, axis=1)
    # onto her skin (+ lift) where the track left the base palm; the normal = her smooth normal there
    tp, tn = fix1_project(surfs["torso_lsh"], tgt, VF, cfg.lift_m, Jf, sigma=5.0)
    wproj = np.clip(moved / 0.02, 0.0, 1.0)
    tgt = tgt + wproj[:, None] * (tp - tgt)
    nrm = _unit(wproj[:, None] * tn + (1 - wproj[:, None]) * Nc[:, 0], tn)
    if len(ci) > 2:
        tl_ = np.einsum("kji,kj->ki", Fr, tgt - Jf[:, SPINE3])
        tl_[ci] = speed_limit_track(tl_[ci], cfg.rate_mps / HZ)
        tgt = np.einsum("kij,kj->ki", Fr, tl_) + Jf[:, SPINE3]
    fd = fdir_for(tgt, nrm)
    if W > 3:
        # the fingers' direction smoothed in her frame (the hand never spins: its fingertips are what the speed audit sees first)
        fdl = gaussian_filter1d(np.einsum("kji,kj->ki", Fr, fd), 5.0, axis=0, mode="nearest")
        fd = _unit(np.einsum("kij,kj->ki", Fr, fdl), fd)
    # ---- the elbow's swivel: the IK with the hint, then the elbow turned about the shoulder-wrist axis; per swivel the depth of his forearm /
    # upper arm (his exported skin) in her (her exported skin: torso, collars, upper arms giving upper_arm_allow_m), his hand in her neck,
    # the COCO elbow, the counterbalance sag
    ik0 = fix1_ik(J, R, R_in14, rest, tgt, nrm, fd, hint, cfg)
    S1, W1, E1 = ik0["S"], ik0["W"], ik0["E"]
    u = _unit(W1 - S1)
    Cl = S1 + ((E1 - S1) * u).sum(1, keepdims=True) * u
    # the swivel's reference side: the hint's elbow where the arm is bent; near a straight arm (its elbow on the shoulder-wrist line, the
    # side noisy) "outward and up" (her left + up, across the line) - blended by the elbow's offset, smoothed in time (no side flips)
    ref_out = _unit(xh + yu)
    ref_out = _unit(ref_out - (ref_out * u).sum(1, keepdims=True) * u)
    off = E1 - Cl
    on_ = np.linalg.norm(off, axis=1)
    wbent = np.clip((on_ - 0.03) / 0.05, 0.0, 1.0)[:, None]
    qref = _unit(wbent * _unit(off, ref_out) + (1 - wbent) * ref_out, ref_out)
    if W > 3:
        qref = gaussian_filter1d(qref, 3.0, axis=0, mode="nearest")
        qref = _unit(qref - (qref * u).sum(1, keepdims=True) * u, ref_out)
    E1 = Cl + qref * np.maximum(on_, 0.05)[:, None]
    phis = np.radians(np.asarray(cfg.swivels_deg, float))
    M = len(phis)
    sw_cost = np.zeros((W, M))
    sw_tw = np.zeros((W, M, 3))
    for mi, ph in enumerate(phis):
        Eh = Cl + np.einsum("wij,wj->wi", _rot_about(u, np.full(W, ph)), E1 - Cl)
        ikm = fix1_ik(J, R, R_in14, rest, tgt, nrm, fd, Eh, cfg)
        om = posed(fit_lead, ks, fix1_pose(Pb, ikm))
        VA = ctx["skinL"].verts(om["J"], om["R"], arm_ids)
        da_m = surface_depth(surfs["torso_arms"], VA, VF, vallow=vall_arm, exact=False).max(1)
        VH = ctx["skinL"].verts(om["J"], om["R"], hand_sub)
        dn_m = surface_depth(surfs["neck"], VH, VF, exact=False).max(1)
        pen_cm = (da_m + 2.0 * dn_m) * 100
        sw_cost[:, mi] = cfg.w_pen * 0.01 * pen_cm ** 2 + cfg.w_swivel * abs(ph)
        sag = elbow_sag(ikm["S"], ikm["E"], ikm["W"])
        # the forearm's roll: near a straight arm the elbow's side (the swivel) sets the whole arm's roll - the side that leaves the wrist
        # and the elbow the least twist
        sw_cost[:, mi] += cfg.w_twist_sw * (np.maximum(0.0, np.abs(ikm["twist_deg"]) - 45.0) + 0.6 * np.maximum(0.0, np.abs(ikm["roll_deg"]) - 45.0))
        sw_tw[:, mi] = np.stack([ikm["twist_deg"], ikm["roll_deg"], pen_cm], 1)
        sw_cost[:, mi] += np.where(cls == "cb", cfg.w_sag * np.maximum(0.0, sag - cfg.sag_free_m), 0.0)
        if okE.any():
            sw_cost[:, mi] += np.where(okE, cfg.w_coco_elbow * np.linalg.norm(ikm["E"] - np.nan_to_num(Eo), axis=1), 0.0)
    trs = 15.0 * np.abs(phis[:, None] - phis[None])
    msel = viterbi_tr(sw_cost, np.ones((W, M), bool), trs)
    phi = phis[msel]
    if W > 3:
        phi = gaussian_filter1d(phi, cfg.sigma_frames, mode="nearest")
    # the blend base -> solved: in the TARGET's terms (the hand's facing, its fingers and the elbow hint go from the base arm's to the
    # solved ones by pw, the palm follows the track - round her surface, never through her as a blend of two unlike arms' joint angles
    # would), the joint angles blended (wm) only over the first small move; pw: the move (a move under move_full_m does not reconfigure the
    # arm), never faster than full in ~0.7 s, on the ramps faded
    pw = np.clip(moved / cfg.move_full_m, 0.0, 1.0)
    if W > 3:
        pw = limit_steps(pw[:, None], 1.0 / 20.0, iters=2000)[:, 0]
        pw = np.clip(gaussian_filter1d(pw, 3.0, mode="nearest"), 0.0, 1.0)
    pw = np.minimum(pw, wr)
    wm = np.clip(pw / 0.25, 0.0, 1.0)
    if W > 3:
        wm = np.clip(gaussian_filter1d(wm, 2.0, mode="nearest"), 0.0, 1.0)
    wm = np.where(pw > 0, wm, 0.0)
    act = wm > 0.0
    nb_ = _unit(-hc.palm_normal(J, "right"))
    fb_ = _unit(hc.finger_dir(J, "right"))
    Eb_ = J[:, ELBOW["right"]]

    def hand_frame(n_, f_):
        n_ = _unit(n_)
        f_ = _unit(f_ - (f_ * n_).sum(1, keepdims=True) * n_)
        return np.stack([n_, f_, np.cross(n_, f_)], -1)
    from scipy.spatial.transform import Rotation as Ro_
    Fb_ = hand_frame(nb_, fb_)

    def compose(tgt_, nrm_, fd_, phi_):
        Eh_ = Cl + np.einsum("wij,wj->wi", _rot_about(u, phi_), E1 - Cl)
        p_ = pw[:, None]
        # the hand's facing / fingers from the base's to the target's by a rotation (slerp): a vector blend of a palm on her front facing
        # away from her and a palm on her back passes through zero (the hand flips edge-on into her half-way)
        rv_ = Ro_.from_matrix(np.transpose(Fb_, (0, 2, 1)) @ hand_frame(nrm_, fd_)).as_rotvec()
        Fp_ = Fb_ @ rotvec_to_mat(rv_ * p_)
        ik_ = fix1_ik(J, R, R_in14, rest, tgt_, Fp_[:, :, 0], Fp_[:, :, 1], p_ * Eh_ + (1 - p_) * Eb_, cfg)
        P_ik_ = fix1_pose(Pb, ik_)
        P_ik_[:, 40:55] = np.stack([_slerp_rotvec(Pb[:, 40 + i], fing[:, i], pw) for i in range(15)], 1)
        if W > 5:
            P_ik_ = smooth_rotations(P_ik_, (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"]), 1.5)
        P_ = Pb.copy()
        for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
            P_[:, j] = _slerp_rotvec(Pb[:, j], P_ik_[:, j], wm)
        P_[~act] = Pb[~act]
        return P_, ik_, P_ik_
    # ---- the frames the arm cannot reach (the leans: the straight arm short of her back): the target laid onto her surface where the arm's
    # palm ends up (+ lift), twice - the palm on her, not in her nor in the air
    P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
    for _ in range(2):
        far = act & (ikf["residual_m"] > 0.01)
        if not far.any():
            break
        o_ = posed(fit_lead, ks, P_ik)
        pr = hc.palm_centre(o_["J"], "right")
        tp_, tn_ = fix1_project(surfs["torso_lsh"], pr, VF, cfg.lift_m, Jf)
        wf = far.astype(float)
        if W > 3:
            wf = np.clip(gaussian_filter1d(wf, 2.0, mode="nearest") * 1.5, 0.0, 1.0)
        tgt = tgt + wf[:, None] * (tp_ - tgt)
        nrm = _unit(wf[:, None] * tn_ + (1 - wf[:, None]) * nrm, nrm)
        P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
    # ---- the hand laid on her: her surface curves under the flat hand (the thumb, the heel, the fingertips sink in where her back is hollow
    # between the blade and the spine): per frame the hand tilted / rolled a little about the palm centre (a rigid search on his skin's
    # hand) and lifted just enough, the lift <= 3 cm and only where the arm reaches; twice
    from scipy.ndimage import maximum_filter1d
    tilts = [(a_, b_) for a_ in (-10.0, 0.0, 10.0, 20.0) for b_ in (-8.0, 0.0, 8.0)]
    lift_tot = np.zeros(W)
    rep_lift = []
    P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
    dbg_tw = [ikf["twist_deg"].copy()]
    dbg_phi0 = phis[msel]
    # a wrist twisted past 80 deg (the palm flush on her asks more pronation / supination than the forearm gives): the hand turned about
    # the forearm by the least of +-20 / 40 / 60 deg that brings the twist down (the palm a little less flush); before and after the lift

    def untwist(nrm_, fd_, ikf_):
        twb = (np.abs(ikf_["twist_deg"]) > 80.0) & act
        if not twb.any():
            return nrm_, fd_, False
        fa = _unit(ikf_["W"] - ikf_["E"])
        best_d = np.zeros(W)
        best_tw = np.abs(ikf_["twist_deg"])
        for dd in (-60.0, -40.0, -20.0, 20.0, 40.0, 60.0):
            Rr = _rot_about(fa, np.full(W, np.radians(dd)))
            ikt = fix1_ik(J, R, R_in14, rest, tgt, np.einsum("kij,kj->ki", Rr, nrm_), np.einsum("kij,kj->ki", Rr, fd_),
                          Cl + np.einsum("wij,wj->wi", _rot_about(u, phi), E1 - Cl), cfg)
            tw_ = np.abs(ikt["twist_deg"]) + 0.15 * abs(dd)
            bt = twb & (tw_ < best_tw)
            best_d = np.where(bt, dd, best_d)
            best_tw = np.where(bt, tw_, best_tw)
        if W > 3:
            best_d = gaussian_filter1d(best_d, 3.0, mode="nearest")
        Rr = _rot_about(fa, np.radians(best_d))
        return _unit(np.einsum("kij,kj->ki", Rr, nrm_)), _unit(np.einsum("kij,kj->ki", Rr, fd_)), True
    nrm, fd, chg = untwist(nrm, fd, ikf)
    if chg:
        P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
    dbg_tw.append(ikf["twist_deg"].copy())
    for it in range(2):
        ka = np.flatnonzero(act & (ikf["residual_m"] < 0.01))
        if not len(ka):
            break
        o_ = posed(fit_lead, ks[ka], P_ik[ka])
        VH = ctx["skinL"].verts(o_["J"], o_["R"], hand_sub).astype(float)
        pcx = hc.palm_centre(o_["J"], "right")
        f_ = _unit(hc.finger_dir(o_["J"], "right"))
        v_ = _unit(hc.palm_normal(o_["J"], "right"))
        l_ = _unit(np.cross(f_, v_))
        best_L = np.full(len(ka), np.inf)
        best_t = np.zeros((len(ka), 2))
        for a_, b_ in tilts:
            Rt = _rot_about(l_, np.full(len(ka), np.radians(a_))) @ _rot_about(f_, np.full(len(ka), np.radians(b_)))
            Vt = pcx[:, None] + np.einsum("kij,knj->kni", Rt, VH - pcx[:, None])
            d_ = surface_depth(surfs["torso_arms"], Vt, VF[ka], vallow=vall_h).max(1)
            L_ = np.maximum(0.0, d_ - 0.002) + 0.004 * (abs(a_) + abs(b_)) / 10.0
            bt = L_ < best_L
            best_L = np.where(bt, L_, best_L)
            best_t[bt] = (a_, b_)
        lift = np.zeros(W)
        lift[ka] = np.minimum(np.maximum(0.0, 0.03 - lift_tot[ka]), np.maximum(0.0, best_L))
        ta, tb = np.zeros(W), np.zeros(W)
        ta[ka], tb[ka] = best_t[:, 0], best_t[:, 1]
        if W > 3:
            lift = gaussian_filter1d(maximum_filter1d(lift, 5, mode="nearest"), 1.5, mode="nearest")
            ta, tb = gaussian_filter1d(ta, 4.0, mode="nearest"), gaussian_filter1d(tb, 4.0, mode="nearest")
        lift *= act
        # the tilts applied to the target's normal / finger direction (the IK lays the palm against -normal, the fingers along fdir)
        oo = posed(fit_lead, ks, P_ik)
        fW = _unit(hc.finger_dir(oo["J"], "right"))
        vW = _unit(hc.palm_normal(oo["J"], "right"))
        lW = _unit(np.cross(fW, vW))
        Rt = _rot_about(lW, np.radians(ta)) @ _rot_about(fW, np.radians(tb))
        nrm = _unit(np.einsum("kij,kj->ki", Rt, nrm))
        fd = _unit(np.einsum("kij,kj->ki", Rt, fd))
        tgt = tgt + lift[:, None] * nrm
        lift_tot += lift
        P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
        dbg_tw.append(ikf["twist_deg"].copy())
        rep_lift.append({"lift_cm_max": round(float(lift_tot.max()) * 100, 2), "lift_cm_p90": round(float(np.percentile(lift_tot[act], 90)) * 100, 2)
                         if act.any() else 0.0, "tilt_deg_p90": round(float(np.percentile(np.abs(ta[act]), 90)), 1) if act.any() else 0.0})
    nrm, fd, chg = untwist(nrm, fd, ikf)
    if chg:
        P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
        dbg_tw.append(ikf["twist_deg"].copy())
    # ---- the hand out of her: on the ramps the hand turns round her side (from the base's hand on her front to the palm on her back) and
    # its fingertips cut into her where the turn passes (the lift above covers <= 3 cm, on the reached frames): per frame the deepest point
    # of his hand's skin (the final, blended pose) in her torso / collars / upper arms / neck / head and the target moved along that point's
    # way out by its depth + push_margin_m, dilated and smoothed in time (slow: the speed rules), <= push_max_m in all; push_rounds; then the
    # speed rules inside the window (the input's speeds: the flagged steps low-passed where the window changes the arm) - and once more
    # where that low-pass put the hand back in her
    push_tot = np.zeros((W, 3))
    rep_push = []
    hand_all = ids["hand"][::2]
    rep_speed = None
    for outer in range(2):
        for _ in range(cfg.push_rounds):
            o_ = posed(fit_lead, ks, P_new)
            dep, way = hand_way_out(surfs, ctx["skinL"].verts(o_["J"], o_["R"], hand_all), VF, axes={"neck": (Jf[:, NECK], Jf[:, HEAD])})
            bad = act & (dep > cfg.push_tol_m)
            rep_push.append({"frames": int(bad.sum()), "depth_cm_max": round(float(dep[act].max()) * 100, 2) if act.any() else 0.0})
            if not bad.any():
                break
            need0 = np.where(bad, dep + cfg.push_margin_m, 0.0)
            need, way_s = need0, np.where(bad[:, None], way, 0.0)
            if W > 3:
                need = gaussian_filter1d(maximum_filter1d(need0, 11, mode="nearest"), 3.0, mode="nearest")
                # the frames round a bad one take its way: the ways weighted by the depth, spread in time, normalized (none far from one)
                way_s = gaussian_filter1d(need0[:, None] * way, 4.0, axis=0, mode="nearest")
                wl_ = np.linalg.norm(way_s, axis=1)
                way_s = np.where((wl_ > 1e-5)[:, None], way_s / np.maximum(wl_, 1e-12)[:, None], 0.0)
            tot = push_tot + need[:, None] * way_s
            tot *= np.minimum(1.0, cfg.push_max_m / np.maximum(np.linalg.norm(tot, axis=1), 1e-9))[:, None]
            tgt = tgt + (tot - push_tot)
            push_tot = tot
            P_new, ikf, P_ik = compose(tgt, nrm, fd, phi)
        if ctx.get("p99") is None:
            break
        P_new, rep_speed = speed_repair(ctx, P_new, rounds=8, half=8, sigma=3.0, ks=ks, allowed=act)
        o_ = posed(fit_lead, ks, P_new)
        dep, _ = hand_way_out(surfs, ctx["skinL"].verts(o_["J"], o_["R"], hand_all), VF)
        rep_push.append({"after_speed_repair": int((act & (dep > cfg.push_tol_m)).sum())})
        if int((act & (dep > 0.02)).sum()) <= 2:
            break
    sel_names = np.where(sel == 0, "keep", np.array(grid["names"])[np.maximum(sel - 1, 0)])
    if ctx.get("debug_dir"):
        np.savez_compressed(Path(ctx["debug_dir"]) / f"fix1_win_{c0}_{c1}.npz", ks=ks, sel=sel, sel_names=sel_names.astype(str), cost=cost,
                            allowed=allowed, tgt=tgt, nrm=nrm, fd=fd, phi=phi, wm=wm, lift=lift_tot, miss=ikf["residual_m"],
                            twist=ikf["twist_deg"], roll=ikf["roll_deg"], bend=ikf["bend_deg"], collar=ikf["collar_deg"], Xc=Xc,
                            theta_f=grid["theta_f"], dy_f=grid["dy_f"], sw_cost=sw_cost, reach=reach, sw_tw=sw_tw, dbg_tw=np.stack(dbg_tw, 1), dbg_phi0=dbg_phi0)
    stats = {"frames": int(W), "core": [int(c0), int(c1)], "classes": {c_: int((cls[core] == c_).sum()) for c_ in ("ulb", "cb", "back", "")},
             "base_ok_core": int(ok[core].sum()), "kept_core": int((sel[core] == 0).sum()),
             "selected": {str(k_): int(v_) for k_, v_ in zip(*np.unique(sel_names[core].astype(str), return_counts=True))},
             "reach_ratio_p50_max": [round(float(np.nanmedian(reach[core, sel[core]])), 3) if (sel[core] > 0).any() else None,
                                     round(float(np.nanmax(reach[core, sel[core]])), 3) if (sel[core] > 0).any() else None],
             "ik_miss_cm_max": round(float(ikf["residual_m"][core].max()) * 100, 2), "collar_deg_max": round(float(ikf["collar_deg"].max()), 2),
             "move_cm_max": round(float(moved.max()) * 100, 1), "coco_elbow_frames": int(okE.sum()),
             "swivel_deg_p50": round(float(np.degrees(np.median(np.abs(phi[core])))), 1), "lift_rounds": rep_lift, "push_rounds": rep_push,
             "push_cm_max": round(float(np.linalg.norm(push_tot, axis=1).max()) * 100, 2), "speed_repair": rep_speed}
    return {"ks": ks, "P": P_new, "target": tgt, "normal": nrm, "sel": sel_names, "phi": phi, "w": wm, "stats": stats, "VF": VF, "kf": kf}


FIX1_PUSH_DIRS = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0), (0.7, 0.0, 0.7), (0.7, -0.7, 0.0), (0.0, -0.7, 0.7),
                  (0.7, 0.0, -0.7), (0.6, -0.5, 0.6))     # fix1_pushout's ways for his wrist, in her torso frame (+ toward his shoulder)
FIX1_PUSH_MAGS = (0.02, 0.04, 0.06, 0.08, 0.10)


def fix1_pushout(ctx: dict, win: dict, cfg: "Fix1Cfg", rounds: int = 3, neck_tol_m: float = 0.004, tol_m: float = 0.012) -> dict:
    """the fallback of a window no re-pose passes whose base has his hand in her neck / head (the verifier's F7: his wrist / thumb inside her
    neck where the hand reaches over the top of her shoulder): the base kept, only his hand moved out of her - on the frames whose hand is
    deeper than neck_tol_m in her neck / head, the wrist moved by the best of FIX1_PUSH_DIRS (her frame) + "toward his shoulder" x
    FIX1_PUSH_MAGS (cost: 6 / cm in her neck, 6 / cm in her torso beyond tol_m, 3 / cm the wrist misses its place, 0.15 / cm moved, 0.5 /
    cm of the palm in front of her beyond +4 cm, 2 / cm of counterbalance elbow sag beyond 8 cm), the hand's global rotation kept and the
    arm re-solved (arm_keep_hand); the moves (<= 10 cm) spread to the frames round (+-12) and smoothed slowly (the speed rules); rounds;
    then the speed rules (kept only where they do not put the hand back in her neck) -> like fix1_window ({"ks", "P", "stats", "VF", "kf"})"""
    from scipy.ndimage import gaussian_filter1d
    fit_lead = ctx["fit_lead"]
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    ks = np.arange(s0, s1 + 1)
    W = len(ks)
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    VF = ctx["skinF"].verts(kf["J"], kf["R"])
    Fr = torso_frame(kf["J"])
    axes = {"neck": (kf["J"][:, NECK], kf["J"][:, HEAD])}
    surfs = ctx["surfs"]
    hand_ids = ctx["ids"]["hand"][::2]
    Pfull = ctx["P_base"]
    o0 = posed(fit_lead, ks, Pfull[ks])
    W0 = o0["J"][:, WRIST["right"]]
    G21 = o0["R"][:, WRIST["right"]]

    cls = ctx["cls"][ks]

    def depths(P_, fr, extra=False):
        o_ = posed(fit_lead, ks[fr], P_)
        VH = ctx["skinL"].verts(o_["J"], o_["R"], hand_ids)
        dn, _ = hand_way_out(surfs, VH, VF[fr], ("neck",), axes={"neck": (axes["neck"][0][fr], axes["neck"][1][fr])})
        dt, _ = hand_way_out(surfs, VH, VF[fr], ("torso_arms",))
        if not extra:
            return dn, dt
        z = in_frame(hc.palm_centre(o_["J"], "right"), kf["J"][fr])[:, 2]
        sag = np.where(cls[fr] == "cb", elbow_sag(o_["J"][:, SHOULDER["right"]], o_["J"][:, ELBOW["right"]], o_["J"][:, WRIST["right"]]), 0.0)
        return dn, dt, z, sag

    def place(shift_):
        P_ = Pfull[ks].copy()
        mv_ = np.linalg.norm(shift_, axis=1) > 1e-5
        if mv_.any():
            Pn, info = arm_keep_hand(fit_lead, Pfull, Pfull, ks[mv_], X_wrist=W0[mv_] + shift_[mv_], G21=G21[mv_])
            P_[mv_] = Pn[ks[mv_]]
        return P_
    shift = np.zeros((W, 3))
    P = Pfull[ks].copy()
    rep = []
    for _ in range(rounds):
        dn, dt, z0, sg0 = depths(P, np.arange(W), extra=True)
        bad = dn > neck_tol_m
        rep.append({"neck_frames": int(bad.sum()), "neck_cm_max": round(float(dn.max()) * 100, 2), "torso_gt2": int((dt > 0.02).sum())})
        if not bad.any():
            break
        fb = np.flatnonzero(bad)
        Sb = posed(fit_lead, ks[fb], P[fb])["J"][:, SHOULDER["right"]]
        Wc = W0[fb] + shift[fb]
        cands = [np.einsum("kij,j->ki", Fr[fb], _unit(np.asarray(d_, float))) for d_ in FIX1_PUSH_DIRS] + [_unit(Sb - Wc)]
        def cost_(dn_, dt_, miss_, sh_, z_, sg_):
            return (6.0 * dn_ + 6.0 * np.maximum(0.0, dt_ - tol_m) + 3.0 * miss_ + 0.15 * np.linalg.norm(sh_, axis=1)
                    + 0.5 * np.maximum(0.0, z_ - 0.04) + 2.0 * np.maximum(0.0, sg_ - 0.08)) * 100
        best_c = cost_(dn[fb], dt[fb], np.zeros(len(fb)), shift[fb], z0[fb], sg0[fb])
        best_s = shift[fb].copy()
        for dv in cands:
            for mg in FIX1_PUSH_MAGS:
                sh_ = shift[fb] + mg * dv
                sh_ *= np.minimum(1.0, 0.10 / np.maximum(np.linalg.norm(sh_, axis=1), 1e-9))[:, None]
                Pn, info = arm_keep_hand(fit_lead, Pfull, Pfull, ks[fb], X_wrist=W0[fb] + sh_, G21=G21[fb])
                dn_c, dt_c, z_c, sg_c = depths(Pn[ks[fb]], fb, extra=True)
                c_ = cost_(dn_c, dt_c, info["wrist_miss_m"], sh_, z_c, sg_c)
                bt = c_ < best_c - 1e-9
                best_c = np.where(bt, c_, best_c)
                best_s[bt] = sh_[bt]
        raw = shift.copy()
        raw[fb] = best_s
        # each frame takes the largest move among the bad frames within +-7 of it (else its own), then smoothed
        mag = np.where(bad, np.linalg.norm(raw, axis=1), 0.0)
        spread = shift.copy()
        for i in range(W):
            lo, hi = max(0, i - 15), min(W, i + 16)
            j = lo + int(mag[lo:hi].argmax())
            if mag[j] > 0:
                spread[i] = raw[j]
        shift = gaussian_filter1d(spread, 6.0, axis=0, mode="nearest") if W > 3 else spread
        P = place(shift)
    rep_speed = None
    mvd = np.abs(P - Pfull[ks]).max((1, 2)) > 0
    if ctx.get("p99") is not None and mvd.any():
        P_s, rep_speed = speed_repair(ctx, P, rounds=8, half=8, sigma=3.0, ks=ks, allowed=mvd)
        dn_a, _ = depths(P, np.arange(W))
        dn_b, _ = depths(P_s, np.arange(W))
        back_in = dn_b > np.maximum(dn_a, 0.008)
        if back_in.any():
            # the low-pass put the hand back in her neck there: those frames (+-6) keep the pushed pose
            from scipy.ndimage import binary_dilation
            keep = binary_dilation(back_in, iterations=6)
            P_s[keep] = P[keep]
            rep_speed["kept_pushed_frames"] = int(keep.sum())
        P = P_s
    dn, dt = depths(P, np.arange(W))
    rep.append({"final_neck_frames": int((dn > 0.01).sum()), "final_neck_cm_max": round(float(dn.max()) * 100, 2)})
    stats = {"frames": int(W), "core": [int(c0), int(c1)], "fallback": "neck", "pushout": rep, "speed_repair": rep_speed,
             "shift_cm_max": round(float(np.linalg.norm(shift, axis=1).max()) * 100, 2), "frames_moved": int(mvd.sum())}
    return {"ks": ks, "P": P, "stats": stats, "VF": VF, "kf": kf}


def arm_points(fit_lead: dict, P: np.ndarray, ks: np.ndarray | None = None) -> dict:
    """his right elbow, wrist, palm centre, five fingertips (n,3) and the elbow's flexion (deg) for the poses P (on ks, or every frame)"""
    o = posed(fit_lead, ks, P)
    out = {"elbow": o["J"][:, ELBOW["right"]], "wrist": o["J"][:, WRIST["right"]], "palm": hc.palm_centre(o["J"], "right")}
    for fi in FINGERS:
        out[f"tip_{fi}"] = o["tips"][:, tip_index("right", fi)]
    out["flex"] = elbow_angles(P, "right")["flexion"]
    return out


def speed_refs(fit_in: dict) -> dict:
    """the input's p99 speeds of his right elbow / wrist / palm / fingertips (m/s) and of the elbow's flexion (deg/s), the whole film"""
    a = arm_points(fit_in["lead"], hc.full_pose(fit_in["lead"]))
    out = {k: float(np.percentile(hc.speeds(v), 99)) for k, v in a.items() if k != "flex"}
    out["flex"] = float(np.percentile(np.abs(np.diff(a["flex"])) * HZ, 99))
    return out


def fix1_speed_flags(a_in: dict, a: dict, p99: dict, flex_margin: float = 30.0) -> dict:
    """per step (n-1,) of the speed rules against the input (a_in, a: arm_points on the same frames): "fast" (any point faster than
    max(p99, input there) + 0.25 m/s), "addfast" (its change against the input faster than 0.55 m/s while faster than the input), "flexfast"
    (the elbow's flexion speed above max(input p99, input there) + flex_margin deg/s)"""
    n = len(a["elbow"]) - 1
    fast, addf = np.zeros(n, bool), np.zeros(n, bool)
    for k in a:
        if k == "flex":
            continue
        v, vi = hc.speeds(a[k]), hc.speeds(a_in[k])
        fast |= v > np.maximum(p99[k], vi) + 0.25
        addf |= (hc.speeds(a[k] - a_in[k]) > 0.55) & (v > vi)
    fl, fli = np.abs(np.diff(a["flex"])) * HZ, np.abs(np.diff(a_in["flex"])) * HZ
    return {"fast": fast, "addfast": addf, "flexfast": fl > np.maximum(p99["flex"], fli) + flex_margin}


def phone_back_view(obs, ks: np.ndarray, Jf: np.ndarray) -> np.ndarray:
    """(W, C) the phone sees her back (her front faces away from it: cos < -0.2 - a side view does not see a hand behind her)"""
    zh = torso_frame(Jf)[:, :, 2]
    out = np.zeros((len(ks), len(obs.cams)), bool)
    for ci, c in enumerate(obs.cams):
        z = obs.z[c]
        Cc = -np.einsum("kji,kj->ki", z["R"][ks], z["t"][ks]) - Jf[:, SPINE3]
        Cc[:, 1] = 0
        out[:, ci] = (Cc * zh).sum(1) / np.maximum(np.linalg.norm(Cc, axis=1), 1e-9) < -0.2
    return out


def fix1_coco(ctx: dict, Ps: dict, ks: np.ndarray, which: str) -> dict:
    """the independent COCO elbow / wrist (median px) of the motions Ps {name: (W,55,3) on ks} over the same phone-frames: the phones that see
    the CANDIDATE's joint (Ps["new"]: joint_visible); a wrist behind her (z < 0 in her torso frame) only from a phone behind her (PLAN H5: no
    front phone vouches for a hand on her back) -> {name: px or None, "n"}"""
    if "obs" not in ctx:
        ctx["obs"] = hc.Obs()
        ctx["judge"] = hc.CocoJudge(list(ctx["obs"].cams))
        ctx["bodies"] = capsule_bodies(ctx["fit_in"])
    obs, judge = ctx["obs"], ctx["judge"]
    fl = ctx["fit_lead"]
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    jj = ELBOW["right"] if which == "elbow" else WRIST["right"]
    kn = posed(fl, ks, Ps["new"])
    vis = joint_visible(obs, ks, kn["J"][:, jj], ctx["bodies"], kn, kf)
    if which == "wrist":
        behind = in_frame(kn["J"][:, jj], kf["J"])[:, 2] < 0.0
        vis &= ~behind[:, None] | phone_back_view(obs, ks, kf["J"])
    Xs = {nm: posed(fl, ks, P)["J"][:, jj] for nm, P in Ps.items()}
    Xl = kn["J"][:, ELBOW["left"] if which == "elbow" else WRIST["left"]]
    out = {"n": 0}
    for nm, X in Xs.items():
        alt = Xs["new"] if nm != "new" else Xs.get("base")
        e = coco_errors_vis(judge, obs, X, ks, which, vis, Xl, alt)
        out[nm] = round(float(np.median(e)), 1) if len(e) else None
        out["n"] = max(out["n"], int(len(e)))
    return out


def fix1_frame_badness(m: dict, sc: dict, cls: np.ndarray) -> np.ndarray:
    """(W,) how bad each frame is (the independent check's measures): on her front 3, off her upper-left back (outside the leans) 2 / in front
    of her coronal plane (the leans) 1, his hand > 2 cm in her 2 (> 5 cm + 4), his forearm > 2 cm in her left upper arm 1.5 / in her 1, his
    hand in her neck 6, a sagging counterbalance elbow 2"""
    cb = np.asarray(cls) == "cb"
    b = 3.0 * m["front"] + np.where(cb, 1.0 * (m["z"] > 0.0), 2.0 * ~m["ulb"])
    b = b + 2.0 * (sc["hand"] > 0.02) + 4.0 * (sc["hand"] > 0.05) + 1.5 * (sc["fore_lua"] > 0.02) + 1.0 * (sc["fore"] > 0.02)
    b = b + 6.0 * (sc["neck"] > 0.01) + 2.0 * (cb & (m["sag"] > 0.10))
    return b * (np.asarray(cls) != "")


def fix1_select(ctx: dict, r: dict, cfg: "Fix1Cfg", min_s: float = 0.5, fill_s: float = 0.4) -> dict:
    """the solved window kept only on the stretches where it is better than the base frame by frame (fix1_frame_badness, runs of at least
    min_s, gaps under fill_s filled), blended in / out so that the palm's way between the two arms never runs faster than blend_mps (raised
    cosine ramps sized by the distance) -> r with "P" the composite (+ "select": the stretches, the weight)"""
    from scipy.ndimage import uniform_filter1d
    ks = r["ks"]
    W = len(ks)
    fl = ctx["fit_lead"]
    cls = ctx["cls"][ks]
    Pb, Pn = ctx["P_base"][ks], r["P"]
    diff = np.abs(Pn - Pb).max((1, 2)) > 0
    bad = {}
    pcs = {}
    for nm, P in (("base", Pb), ("new", Pn)):
        m = fix1_metrics(fl, P, ks, r["kf"], ctx["sites_f"], cls)
        sc = skin_contact(fl, P, ks, r["kf"], r["VF"], ctx["skinL"], ctx["ids"], ctx["surfs"])
        bad[nm] = uniform_filter1d(fix1_frame_badness(m, sc, cls), 9, mode="nearest")
        pcs[nm] = m["pc"]
    better = diff & (bad["new"] < bad["base"] - 0.2)
    # gaps under fill_s inside better stretches filled, stretches under min_s dropped
    nf, nm_ = int(round(fill_s * HZ)), int(round(min_s * HZ))
    for a, b in _runs(~better, 0):
        if 0 < a and b < W - 1 and (b - a + 1) < nf and diff[a:b + 1].all():
            better[a:b + 1] = True
    for a, b in _runs(better, 0):
        if (b - a + 1) < nm_:
            better[a:b + 1] = False
    D = np.linalg.norm(pcs["new"] - pcs["base"], axis=1)
    w = np.zeros(W)
    for a, b in _runs(better, 0):
        w[a:b + 1] = 1.0
        for e, sgn in ((a, -1), (b, 1)):
            L = int(np.ceil(max(D[e], 0.01) / (cfg.blend_mps / HZ) * np.pi / 2))     # a raised cosine peaks at pi/2 x its mean speed
            for i in range(1, L + 1):
                k = e + sgn * i
                if k < 0 or k >= W or not diff[k]:
                    break
                w[k] = max(w[k], 0.5 + 0.5 * np.cos(np.pi * i / (L + 1)))
    P = Pb.copy()
    on = w > 0
    if on.any():
        for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
            P[on, j] = _slerp_rotvec(Pb[on, j], Pn[on, j], w[on])
    r = dict(r)
    r["P"] = P
    r["select"] = {"stretches": [[round(ks[a] / HZ, 2), round(ks[b] / HZ, 2)] for a, b in _runs(better, 0)],
                   "frames_new": int((w >= 0.999).sum()), "frames_blended": int(((w > 0) & (w < 0.999)).sum()), "frames_base": int((w <= 0).sum())}
    return r


def fix1_check(ctx: dict, r: dict, cfg: "Fix1Cfg") -> dict:
    """a solved window against the base and the input on its core / span: where the palm is (the verifier's measures), his hand / forearm in
    her exported skin, her neck, the counterbalance sag, the COCO elbow / wrist, the speed rules, the wrist / elbow limits -> {"ok", "why",
    "numbers"}"""
    ks = r["ks"]
    c0, c1 = r["stats"]["core"]
    core = (ks >= c0) & (ks <= c1)
    cls = ctx["cls"][ks]
    fl = ctx["fit_lead"]
    Pb, Pi, Pn = ctx["P_base"][ks], ctx["P_in"][ks], r["P"]
    num = {}
    m = {nm: fix1_metrics(fl, P, ks, r["kf"], ctx["sites_f"], cls) for nm, P in (("input", Pi), ("base", Pb), ("new", Pn))}
    s = {nm: skin_contact(fl, P, ks, r["kf"], r["VF"], ctx["skinL"], ctx["ids"], ctx["surfs"]) for nm, P in (("input", Pi), ("base", Pb), ("new", Pn))}
    for nm in ("input", "base", "new"):
        num[nm] = {"ulb": round(float(m[nm]["ulb_cb"][core].mean()), 3), "front": round(float(m[nm]["front"][core].mean()), 3),
                   "back": round(float(m[nm]["back"][core].mean()), 3), "z_cm_p50": round(float(np.median(m[nm]["z"][core])) * 100, 1),
                   "cb_sag_gt10": int(((m[nm]["sag"] > 0.10) & (cls == "cb"))[core].sum()),
                   "hand_gt2": int((s[nm]["hand"] > 0.02).sum()), "hand_gt5": int((s[nm]["hand"] > 0.05).sum()),
                   "fore_gt2": int((s[nm]["fore"] > 0.02).sum()), "fore_gt4": int((s[nm]["fore"] > 0.04).sum()),
                   "fore_lua_gt2": int((s[nm]["fore_lua"] > 0.02).sum()), "fore_lua_gt4": int((s[nm]["fore_lua"] > 0.04).sum()),
                   "fore_lua_max_cm": round(float(s[nm]["fore_lua"].max()) * 100, 1),
                   "neck_gt1": int((s[nm]["neck"] > 0.01).sum()), "palm_gap_gt3": int((s[nm]["palm_gap"] > 0.03).sum()),
                   "hand_max_cm": round(float(s[nm]["hand"].max()) * 100, 1), "fore_max_cm": round(float(s[nm]["fore"].max()) * 100, 1)}
    a_in, a_b, a_n = (arm_points(fl, P, ks) for P in (Pi, Pb, Pn))
    fb, fn = fix1_speed_flags(a_in, a_b, ctx["p99"]), fix1_speed_flags(a_in, a_n, ctx["p99"])
    for k_ in ("fast", "addfast", "flexfast"):
        num["base"][k_], num["new"][k_] = int(fb[k_].sum()), int(fn[k_].sum())
    Wb, Wn = wrist_angles(Pb, "right"), wrist_angles(Pn, "right")
    Eb, En = elbow_angles(Pb, "right"), elbow_angles(Pn, "right")
    for nm, Wa, Ea in (("base", Wb, Eb), ("new", Wn, En)):
        num[nm].update({"bend_gt100": int((Wa["bend"] > 100).sum()), "twist_gt90": int(((np.abs(Wa["twist"]) > 90) & (Wa["bend"] < 60)).sum()),
                        "hyper": int((Ea["flexion"] < -5).sum()), "off_hinge": int((np.abs(Ea["off_hinge"]) > 35).sum())})
    ce = fix1_coco(ctx, {"input": Pi[core], "base": Pb[core], "new": Pn[core]}, ks[core], "elbow")
    cw = fix1_coco(ctx, {"input": Pi[core], "base": Pb[core], "new": Pn[core]}, ks[core], "wrist")
    num["coco_elbow_px"], num["coco_wrist_px"] = ce, cw
    why = []
    b, n = num["base"], num["new"]
    nc = int(core.sum())
    # the upper-left back counts outside the counterbalance leans; in them (the verifier's F2: her left flank / back edge where the back of
    # her shoulder is out of reach, never over her shoulder from the front) only "not on her front"
    ncb = core & (cls != "cb")
    for nm in ("input", "base", "new"):
        num[nm]["ulb_not_cb"] = round(float(m[nm]["ulb"][ncb].mean()), 3) if ncb.any() else None
    ub, un = b.get("ulb_not_cb"), n.get("ulb_not_cb")
    better = ((n["front"] < b["front"] - 0.02) or (ub is not None and un > ub + 0.02) or (n["cb_sag_gt10"] < b["cb_sag_gt10"])
              or (n["neck_gt1"] < b["neck_gt1"]) or (n["hand_gt2"] < b["hand_gt2"] - max(3, int(0.05 * nc))))
    if not better:
        why.append(f"no better: front {b['front']} -> {n['front']}, ULB (outside the leans) {ub} -> {un}")
    # the neck fallback (fix1_pushout: the hand only moved out of her neck, the verifier's F7) may move the palm's location a little
    tol_f, tol_u = (0.08, 0.08) if r["stats"].get("fallback") == "neck" else (0.02, 0.05)
    if n["front"] > b["front"] + tol_f or (ub is not None and un < ub - tol_u):
        why.append(f"location worse: front {b['front']} -> {n['front']}, ULB (outside the leans) {ub} -> {un}")
    i_ = num["input"]
    # a move of the palm from her front to her back (the owner's main request, PLAN H5) is worth some contact his arm makes with hers on the
    # way (her raised left arm rests on his in the closed hold and the leans): the guard rails widen for it; her neck never
    moved_back = (b["front"] - n["front"]) >= 0.3
    tol = (lambda frac, lo: max(lo, int(frac * nc)))
    if moved_back:
        lim2, lim5, lim_lua, lim_f4, lim_lim = tol(0.10, 5), tol(0.04, 4), tol(0.12, 5), tol(0.06, 5), tol(0.06, 5)
    else:
        lim2, lim5, lim_lua, lim_f4, lim_lim = tol(0.06, 3), tol(0.02, 3), tol(0.05, 3), tol(0.03, 3), tol(0.03, 3)
    if n["hand_gt2"] > b["hand_gt2"] + lim2 or n["hand_gt5"] > b["hand_gt5"] + lim5:
        why.append(f"hand into her > 2 / 5 cm {b['hand_gt2']} / {b['hand_gt5']} -> {n['hand_gt2']} / {n['hand_gt5']}")
    # his forearm under her raised left upper arm (closed hold: her arm rests on his): a move to her back may press it 2-4 cm on more
    # frames (the fitted meshes do not give way); deeper than 4 cm only on as many frames as before (+ a few), never past 4.5 cm
    lua_ok = (n["fore_lua_gt2"] <= max(b["fore_lua_gt2"], i_["fore_lua_gt2"]) + lim_lua) or (
        moved_back and n["fore_lua_gt2"] <= max(b["fore_lua_gt2"], i_["fore_lua_gt2"]) + tol(0.30, 5)
        and n["fore_lua_gt4"] <= max(b["fore_lua_gt4"], i_["fore_lua_gt4"]) + tol(0.02, 3)
        and n["fore_lua_max_cm"] <= max(4.5, b["fore_lua_max_cm"], i_["fore_lua_max_cm"]))
    if not lua_ok:
        why.append(f"forearm into her left upper arm > 2 cm {i_['fore_lua_gt2']} (input) / {b['fore_lua_gt2']} -> {n['fore_lua_gt2']} "
                   f"(> 4 cm {b['fore_lua_gt4']} -> {n['fore_lua_gt4']}, max {n['fore_lua_max_cm']} cm)")
    if n["fore_gt4"] > max(b["fore_gt4"], i_["fore_gt4"]) + lim_f4:
        why.append(f"forearm into her > 4 cm {i_['fore_gt4']} (input) / {b['fore_gt4']} -> {n['fore_gt4']}")
    if n["neck_gt1"] > min(b["neck_gt1"], max(i_["neck_gt1"], 0)) and n["neck_gt1"] > 0:
        why.append(f"hand in her neck / head > 1 cm {i_['neck_gt1']} (input) / {b['neck_gt1']} -> {n['neck_gt1']}")
    if n["cb_sag_gt10"] > b["cb_sag_gt10"]:
        why.append(f"counterbalance elbow sag > 10 cm {b['cb_sag_gt10']} -> {n['cb_sag_gt10']}")
    # the speed rules: a move from her front to her back turns the hand over on the ramp (its fingertips' change against the input passes
    # 0.55 m/s for ~1 s: the verifier's F10, minor) - the added-speed rule widens for it; the absolute rule ("fast": past the input's p99
    # + 0.25 m/s) does not
    for k_, sl in (("fast", 3), ("addfast", tol(0.30, 5) if moved_back else tol(0.04, 5)), ("flexfast", tol(0.03, 5))):
        if n[k_] > b[k_] + sl:
            why.append(f"speed rule {k_} {b[k_]} -> {n[k_]} steps")
    for k_ in ("bend_gt100", "twist_gt90", "hyper", "off_hinge"):
        if n[k_] > b[k_] + lim_lim:
            why.append(f"{k_} {b[k_]} -> {n[k_]} frames")
    # the COCO elbow allows more for such a move (the elbow goes out and back with the hand), the COCO wrist (the detectors see only the
    # arm's end in front of her) is advisory then
    ref_e = [x for x in (ce.get("base"), ce.get("input")) if x is not None]
    if ce.get("new") is not None and ref_e and ce["n"] >= 8:
        lim = max(ref_e) + (max(10.0, 0.8 * max(ref_e)) if moved_back else max(8.0, 0.3 * max(ref_e)))
        if ce["new"] > lim:
            why.append(f"COCO elbow {ce.get('input')} (input) / {ce.get('base')} (base) -> {ce['new']} px")
    ref_w = [x for x in (cw.get("base"), cw.get("input")) if x is not None]
    num["notes"] = ["COCO wrist advisory (her front -> her back)"] if moved_back else []
    if cw.get("new") is not None and ref_w and cw["n"] >= 8 and not moved_back:
        lim = max(ref_w) + max(10.0, 0.4 * max(ref_w))
        if cw["new"] > lim:
            why.append(f"COCO wrist {cw.get('input')} (input) / {cw.get('base')} (base) -> {cw['new']} px")
    return {"ok": not why, "why": why, "numbers": num}


def fix1_plan(ctx: dict, cfg: "Fix1Cfg", neck: np.ndarray | None = None) -> list[dict]:
    """the fix-1 windows on the base: runs of torso-hold frames whose palm is on her front, off her upper-left back (the counterbalance class
    also the back / top of her left shoulder) or in front of her coronal plane, whose counterbalance elbow sags, or whose hand is in her
    neck (neck: (T,) depth, m), merged (merge_s), padded (pad_s); the cores never on a frame where his right hand holds hers, the ramps
    stopping 3 frames short of one (ramps_clear_of)"""
    T = len(ctx["cls"])
    cls = ctx["cls"]
    m = fix1_metrics(ctx["fit_lead"], ctx["P_base"], None, ctx["kf"], ctx["sites_f"], cls)
    has = cls != ""
    bad = has & (m["front"] | ((cls != "back") & ((m["z"] > 0.0) | ~m["ulb_cb"])) | ((cls == "cb") & (m["sag"] > cfg.sag_bad_m)))
    if neck is not None:
        bad |= has & (np.asarray(neck) > 0.005)
    hh = ctx["names"] == "hand_hold"
    pad = int(round(cfg.pad_s * HZ))
    wins = []
    for a, b in _runs(bad, int(round(cfg.merge_s * HZ))):
        a, b = max(0, a - pad), min(T - 1, b + pad)
        q = np.flatnonzero(has[a:b + 1] & ~hh[a:b + 1])
        if not len(q):
            continue
        a, b = a + int(q[0]), a + int(q[-1])
        # split at hand-hold frames inside
        seg = has[a:b + 1] & ~hh[a:b + 1]
        for a2, b2 in _runs(seg, 0):
            r0, r1 = a + a2, a + b2
            if (r1 - r0 + 1) / HZ < cfg.min_s:
                continue
            # long runs in chunks of <= max_core_s (each kept or not on its own; the chunks are solved in time order, a chunk's ramps blend
            # from the motion with the chunks before it in)
            n = int(np.ceil((r1 - r0 + 1) / (cfg.max_core_s * HZ)))
            edges = np.linspace(r0, r1 + 1, n + 1).round().astype(int)
            for ci in range(n):
                c0, c1 = int(edges[ci]), int(edges[ci + 1]) - 1
                c0, c1, rl, rr, _, _ = ramps_clear_of(c0, c1, cfg.ramp_max, hh, cfg.ramp_max, max(3, int(round(cfg.min_s * HZ))),
                                                      ci == 0, ci == n - 1)
                if c1 - c0 + 1 < 3:
                    continue
                wins.append({"core": [int(c0), int(c1)], "span": [int(c0 - rl), int(c1 + rr)], "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)],
                             "chunk": [ci, n], "bad_frames": int(bad[c0:c1 + 1].sum()),
                             "classes": {c_: int((cls[c0:c1 + 1] == c_).sum()) for c_ in ("ulb", "cb", "back")}})
    return wins


def neck_depth_all(ctx: dict, P: np.ndarray, frames: np.ndarray, batch: int = 64) -> np.ndarray:
    """(T,) his right hand's depth in her neck / head (her exported skin; winding numbers) on `frames` (0 elsewhere)"""
    T = len(P)
    out = np.zeros(T)
    fl = ctx["fit_lead"]
    for a in range(0, len(frames), batch):
        kk = np.asarray(frames[a:a + batch])
        o = posed(fl, kk, P[kk])
        VL = ctx["skinL"].verts(o["J"], o["R"], ctx["ids"]["hand"])
        VF = ctx["skinF"].verts(ctx["kf"]["J"][kk], ctx["kf"]["R"][kk])
        out[kk] = surface_depth(ctx["surfs"]["neck"], VL, VF).max(1)
    return out


def speed_repair(ctx: dict, P: np.ndarray, rounds: int = 6, half: int = 6, sigma: float = 2.0, ks: np.ndarray | None = None,
                 allowed: np.ndarray | None = None, joints: tuple | None = None) -> tuple[np.ndarray, dict]:
    """the review's speed rules on the whole film (fix1_speed_flags against the input): around every flagged step his right arm's rotations
    (collar, shoulder, elbow, wrist, fingers) are low-passed (quaternions, sigma frames, +-half frames) - only on frames that already differ
    from the input (the input's frames stay bit-identical) and are `allowed`; repeated while flags remain; ks: P holds only those frames (a
    window) -> (P, {"before", "after"} flag counts)"""
    P = np.asarray(P, float).copy()
    fl = ctx["fit_lead"]
    P_in = ctx["P_in"] if ks is None else ctx["P_in"][ks]
    T = len(P)
    a_in = arm_points(fl, P_in, ks)
    joints = joints or (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55))
    rep = {}
    for rnd in range(rounds + 1):
        f = fix1_speed_flags(a_in, arm_points(fl, P, ks), ctx["p99"])
        bad = f["fast"] | f["addfast"] | f["flexfast"]
        cnt = {k_: int(v_.sum()) for k_, v_ in f.items()}
        if rnd == 0:
            rep["before"] = cnt
        rep["after"] = cnt
        if not bad.any() or rnd == rounds:
            break
        changed = np.abs(P - P_in).max((1, 2)) > 0
        m = np.zeros(T, bool)
        for k in np.flatnonzero(bad):
            m[max(0, k - half):min(T, k + half + 2)] = True
        m &= changed
        if allowed is not None:
            m &= np.asarray(allowed, bool)
        if not m.any():
            break
        for a, b in _runs(m, 0):
            a0, b0 = max(0, a - 8), min(T - 1, b + 8)
            seg = smooth_rotations(P[a0:b0 + 1], joints, sigma)
            P[a:b + 1] = seg[a - a0:b - a0 + 1]
    rep["rounds"] = rnd
    return P, rep


def hand_depths(ctx: dict, P: np.ndarray, frames: np.ndarray, batch: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """(len(frames),) x 2: his right hand's depth (m) in her torso / collars / upper arms and in her neck / head (her exported skin, winding
    numbers) for the poses P (T,55,3) on `frames`"""
    frames = np.asarray(frames, int)
    dh, dn = np.zeros(len(frames)), np.zeros(len(frames))
    for a in range(0, len(frames), batch):
        ks = frames[a:a + batch]
        kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
        VF = ctx["skinF"].verts(kf["J"], kf["R"])
        o = posed(ctx["fit_lead"], ks, P[ks])
        VH = ctx["skinL"].verts(o["J"], o["R"], ctx["ids"]["hand"])
        dh[a:a + batch] = surface_depth(ctx["surfs"]["torso_arms"], VH, VF).max(1)
        dn[a:a + batch] = surface_depth(ctx["surfs"]["neck"], VH, VF).max(1)
    return dh, dn


def fix1_whips(ctx: dict, P: np.ndarray, keep: np.ndarray, pad: int = 8, tol_m: float = 0.005) -> tuple[np.ndarray, dict]:
    """the speed rules' "fast" / "addfast" steps OUTSIDE the kept fix-1 windows (keep (T,) bool: their spans repaired their own speeds) -
    the collar fix's arm re-solve can whip where the elbow passes straight - low-passed (speed_repair, +-pad frames, only frames that differ
    from the input); a run of repaired frames that puts his hand deeper into her (torso / upper arms past 1.5 cm by more than tol_m, or
    neck / head by more than 3 mm) on any frame is put back -> (P, report)"""
    from scipy.ndimage import binary_dilation
    P = np.asarray(P, float)
    f = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P), ctx["p99"])
    st = f["fast"] | f["addfast"]
    fr = np.zeros(len(P), bool)
    fr[:-1] |= st
    fr[1:] |= st
    m = binary_dilation(fr, iterations=pad) & ~np.asarray(keep, bool) & (np.abs(P - ctx["P_in"]).max((1, 2)) > 0)
    rep = {"steps_outside_windows": int((st & ~np.asarray(keep, bool)[:-1]).sum()), "frames_allowed": int(m.sum())}
    if not m.any():
        return P, rep
    # the collar not low-passed (a low-pass of rotations each within the bound may leave it), the rest of the arm and the hand are
    P2, sp = speed_repair(ctx, P, rounds=8, half=pad, sigma=3.0, allowed=m, joints=(SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)))
    rep["speed_repair"] = sp
    ch = np.flatnonzero(np.abs(P2 - P).max((1, 2)) > 0)
    if len(ch):
        h0, n0 = hand_depths(ctx, P, ch)
        h1, n1 = hand_depths(ctx, P2, ch)
        worse = np.zeros(len(P), bool)
        # deeper where it matters: his hand > 1.5 cm in her torso / upper arms, any depth in her neck / head
        worse[ch] = (h1 > np.maximum(h0, 0.015) + tol_m) | (n1 > np.maximum(n0, 0.003) + 0.003)
        back = 0
        for a, b in _runs(np.isin(np.arange(len(P)), ch), 1):
            if worse[a:b + 1].any():
                P2[a:b + 1] = P[a:b + 1]
                back += b - a + 1
        rep.update({"frames_changed": int(len(ch)), "frames_put_back": int(back)})
    f2 = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P2), ctx["p99"])
    rep["flags_after"] = {k_: int(v_.sum()) for k_, v_ in f2.items()}
    return P2, rep


def fix1_groups(wins: list, gap_frames: int = 1) -> list:
    """(len(wins),) group ids: windows whose spans overlap (or come within gap_frames) share a group (they are solved in time order, one
    after the other; groups are independent and may run in parallel)"""
    g, out = 0, []
    for i, w in enumerate(wins):
        if i and w["span"][0] - wins[i - 1]["span"][1] > gap_frames:
            g += 1
        out.append(g)
    return out


def fix1_pass(base: str = "v43", cfg: "Fix1Cfg | None" = None, write: bool = True, only: list | None = None, device: str = "cuda",
              stage: str = FIX1_STAGE, part: tuple | None = None) -> dict:
    """fix round 1 (see the section): the base fit (v43) with the collar bounded against the input and the arm re-solved to keep the hand
    (collar_fix), then every fix-1 window (fix1_plan) re-posed (fix1_window, FIX1_VARIANTS in turn) and kept where fix1_check passes (in
    time order, each window starting from the motion with the earlier windows in it), then the speed repair -> runs/<stage>/
    (poses_merged.npz, refine.json with the kept windows, report.json); write: the base frozen as v43f, the fit v4/smplx_contact_v43
    (curation_v43.json stage <stage> on v43f). part (i, n): only the window groups g with g % n == i (fix1_groups) -> runs/<stage>/
    poses_part<i>.npz + report_part<i>.json, no speed repair / write (fix1_merge joins the parts)"""
    cfg = cfg or Fix1Cfg()
    t0 = time.time()
    ctx = fix1_context(base, device)
    ctx["p99"] = speed_refs(ctx["fit_in"])
    torso = np.flatnonzero(ctx["cls"] != "")
    neck = neck_depth_all(ctx, ctx["P_base"], torso)
    wins = fix1_plan(ctx, cfg, neck)
    groups = fix1_groups(wins)
    log(f"fix1: {len(wins)} windows in {max(groups) + 1 if groups else 0} groups ({time.time() - t0:.0f} s); collar {ctx['collar_report']}")
    rep = {"what": "v43 fix round 1 (zouk_contacts.fix1_pass)", "base": base, "cfg": asdict(cfg), "collar": ctx["collar_report"],
           "p99_input": ctx["p99"], "windows": [], "part": list(part) if part else None}
    def solve(wi, win, sweep):
        win = dict(win)
        t1 = time.time()
        # the configurations in turn (FIX1_VARIANTS over cfg) until one passes the check; else the base stays
        tried = []
        for vi, var in enumerate(FIX1_VARIANTS):
            cfg_v = Fix1Cfg(**{**asdict(cfg), **var})
            r = fix1_window(ctx, win, cfg_v)
            if cfg_v.select:
                r = fix1_select(ctx, r, cfg_v)
            chk = fix1_check(ctx, r, cfg_v)
            tried.append({"variant": vi, "why": chk["why"], "front_ulb": [chk["numbers"]["new"]["front"], chk["numbers"]["new"]["ulb"]]})
            if chk["ok"]:
                break
        if not chk["ok"] and chk["numbers"]["base"]["neck_gt1"] > 0:
            # no re-pose passes and the base has his hand in her neck: the base with only the hand moved out of her (fix1_pushout)
            r = fix1_pushout(ctx, win, cfg)
            chk = fix1_check(ctx, r, cfg)
            vi = "neck"
            tried.append({"variant": vi, "why": chk["why"], "front_ulb": [chk["numbers"]["new"]["front"], chk["numbers"]["new"]["ulb"]]})
        res = {"window": wi, "group": groups[wi], "sweep": sweep, "t": win["t"], "core": win["core"], "span": win["span"], "stats": r["stats"],
               "select": r.get("select"), "check": chk["numbers"], "adopted": bool(chk["ok"]), "why": chk["why"], "variant": vi,
               "tried": tried, "secs": round(time.time() - t1, 1)}
        if chk["ok"]:
            ctx["P_base"][r["ks"]] = r["P"]
        n = chk["numbers"]
        verdict = f"KEPT (variant {vi})" if chk["ok"] else "vetoed"
        whys = "" if chk["ok"] else str([t_["why"] for t_ in tried])
        log(f"fix1 window {wi + 1}/{len(wins)} {win['t']} (sweep {sweep}): {verdict} front {n['base']['front']} -> {n['new']['front']}, "
            f"ULB {n['base']['ulb']} -> {n['new']['ulb']} {whys} ({res['secs']} s)")
        return res
    done = {}
    for wi, win in enumerate(wins):
        if only and not any(win["t"][0] - 1 <= float(t) <= win["t"][1] + 1 for t in only):
            continue
        if part and groups[wi] % int(part[1]) != int(part[0]):
            continue
        done[wi] = solve(wi, win, 0)
    # a second sweep: a vetoed window next to a kept one (the same group: their spans touch) once more - its ramp now starts from the kept
    # window's motion (a chunk of a long run between two kept chunks no longer goes round her to her front and back)
    for wi in sorted(done):
        if done[wi]["adopted"]:
            continue
        nb = [j for j in (wi - 1, wi + 1) if j in done and done[j]["adopted"] and groups[j] == groups[wi]]
        if nb:
            r2 = solve(wi, wins[wi], 1)
            r2["sweep0_why"] = done[wi]["why"]
            done[wi] = r2
    rep["windows"] = [done[k] for k in sorted(done)]
    d = OUT / "runs" / stage
    d.mkdir(parents=True, exist_ok=True)
    if part:
        np.savez_compressed(d / f"poses_part{int(part[0])}.npz", lead=ctx["P_base"])
        rep["seconds"] = round(time.time() - t0, 1)
        write_json(d / f"report_part{int(part[0])}.json", rep)
        return rep
    return fix1_finish(ctx, rep, base, stage, write, only, t0)


def fix1_finish(ctx: dict, rep: dict, base: str, stage: str, write: bool, only=None, t0: float | None = None,
                global_speed: bool = False) -> dict:
    """the collar bounded again, runs/<stage>/ (poses_merged.npz, refine.json with the kept windows, report.json), and (write) fix1_write.
    global_speed: first the speed repair over the whole film (OFF: round f showed its low-pass of the arm round every flagged step on the
    whole film put his hand into her neck on 27 more frames > 2 cm and > 5 cm into her on 82 more - the windows repair their own speeds;
    instead fix1_whips: the fast steps outside the kept windows only, with a depth guard)"""
    t0 = t0 or time.time()
    P = ctx["P_base"]
    if global_speed:
        P, sp = speed_repair(ctx, P)
        rep["speed_repair"] = sp
    else:
        # only the whips outside the kept windows (the collar fix's re-solve), with a depth guard (fix1_whips)
        keep = np.zeros(len(P), bool)
        for w in rep["windows"]:
            if w["adopted"]:
                keep[w["span"][0]:w["span"][1] + 1] = True
        # outside the kept windows the current collar fix (its wrist-twist share smoothed in time; the parts may carry an older one)
        Pc, _ = collar_fix(ctx["fit_in"]["lead"], ctx["fit_lead"])
        P = np.asarray(P, float).copy()
        P_old = P.copy()
        P[~keep] = Pc[~keep]
        # a kept window was solved from the parts' base: the 12 frames before / after its span go from the current collar fix to that base
        # (slerp), so the window's ramp starts where it was solved from
        T_ = len(P)
        for a_, b_ in _runs(keep, 0):
            for idx in (np.arange(max(0, a_ - 12), a_), np.arange(b_ + 1, min(T_, b_ + 13))):
                if not len(idx):
                    continue
                d_ = np.minimum(np.abs(idx - a_), np.abs(idx - b_))
                w_ = np.clip(1.0 - d_ / 13.0, 0.0, 1.0)
                w_ = w_ * w_ * (3 - 2 * w_)
                for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"]):
                    df_ = np.abs(Pc[idx, j] - P_old[idx, j]).max(1) > 0            # (frames that agree stay bit-identical)
                    if df_.any():
                        P[idx[df_], j] = _slerp_rotvec(Pc[idx[df_], j], P_old[idx[df_], j], w_[df_])
        rep["collar_fix_outside_windows_frames_changed"] = int((np.abs(P - ctx["P_base"]).max((1, 2)) > 0).sum())
        P, rep["whips"] = fix1_whips(ctx, P, keep)
        f = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P), ctx["p99"])
        rep["speed_flags_final"] = {k_: int(v_.sum()) for k_, v_ in f.items()}
    # the windows' smoothing (or the speed repair's low-pass) may carry the collar a little past the bound against the input (a smoothed
    # sequence of rotations each near the input's is not each near it): bounded again, the arm re-solved to keep the hand (collar_fix's way)
    P, rep["collar_reclamp"] = collar_reclamp(ctx["fit_lead"], ctx["P_in"], P)
    P_b0 = hc.full_pose(load_fit(base)["lead"])
    rep["frames_changed_vs_base"] = int((np.abs(P - P_b0).max((1, 2)) > 0).sum())
    rep["collar_deg_max_vs_input"] = round(float(collar_change_deg(ctx["P_in"], P).max()), 3)
    rep["seconds"] = round(time.time() - t0, 1)
    d = OUT / "runs" / stage
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / ("poses_merged.npz" if not only else "trial_poses.npz"), lead=P)
    if only:
        write_json(d / "trial.json", rep)
        return rep
    kept = [{"window": w["window"], "adopted": True, "core": list(w["core"]), "ramp": 0, "ramp_l": int(w["core"][0] - w["span"][0]),
             "ramp_r": int(w["span"][1] - w["core"][1]), "hold": "fix1", "t": w["t"]} for w in rep["windows"] if w["adopted"]]
    write_json(d / "refine.json", {"what": "the v43 fix round 1's kept windows (zouk_contacts.fix1_pass)", "results": kept})
    write_json(d / "report.json", rep)
    log(f"fix1: {sum(w['adopted'] for w in rep['windows'])}/{len(rep['windows'])} windows kept, {rep['frames_changed_vs_base']} frames changed "
        f"({rep['seconds']} s)")
    if write:
        rep["written"] = fix1_write(stage, base)
        write_json(d / "report.json", rep)
    return rep


def fix1_merge(n: int, base: str = "v43", stage: str = FIX1_STAGE, write: bool = True, device: str = "cuda") -> dict:
    """the parts of a parallel fix-1 pass (fix1_pass part=(i, n)) joined: each part changed only its own window groups' frames against the
    collar-fixed base (no two parts touch a frame: the groups' spans do not overlap) -> fix1_finish (refine.json, report, write). The base is
    the parts' own (their per-frame median: on every frame at least n - 1 parts hold it), not this machine's collar_fix - a merge on another
    machine (its float rounding) would otherwise see every collar-fixed frame as changed by every part"""
    t0 = time.time()
    ctx = fix1_context(base, device)
    ctx["p99"] = speed_refs(ctx["fit_in"])
    d = OUT / "runs" / stage
    Ps = [np.load(d / f"poses_part{i}.npz")["lead"] for i in range(n)]
    P0 = np.median(np.stack(Ps), 0) if n > 2 else ctx["P_base"].copy()
    rep = {"what": "v43 fix round 1 (zouk_contacts.fix1_pass, parallel parts joined by fix1_merge)", "base": base, "parts": n,
           "collar": ctx["collar_report"], "p99_input": ctx["p99"], "windows": [],
           "base_vs_this_machine_collar_fix_max_abs": float(np.abs(P0 - ctx["P_base"]).max())}
    ctx["P_base"] = P0.copy()
    owner = np.full(len(P0), -1)
    for i in range(n):
        Pi = Ps[i]
        ch = np.abs(Pi - P0).max((1, 2)) > 0
        clash = ch & (owner >= 0)
        if clash.any():
            # never expected (the groups' spans do not overlap): the earlier part's frames stay, reported
            log(f"fix1_merge: WARNING part {i} changes {int(clash.sum())} frames another part changed - those keep the earlier part's")
            rep.setdefault("clashes", []).append({"part": i, "frames": int(clash.sum())})
            ch = ch & ~clash
        owner[ch] = i
        ctx["P_base"][ch] = Pi[ch]
        ri = json.loads((d / f"report_part{i}.json").read_text(encoding="utf-8"))
        rep["windows"] += ri["windows"]
        rep.setdefault("cfg", ri.get("cfg"))
    rep["windows"] = sorted(rep["windows"], key=lambda w: w["window"])
    return fix1_finish(ctx, rep, base, stage, write, None, t0)


def fix1_write(stage: str = FIX1_STAGE, base: str = "v43", frozen: str = "v43f") -> dict:
    """the fix-1 result (runs/<stage>/poses_merged.npz) written as the fit v43: the base (the v43 fit before the fix) frozen as `frozen`
    (freeze_v43: its folder, exact / measure copies, its curation stages), then v4/smplx_contact_v43 = the frozen fit with the new leader
    poses, curation_v43.json stage <stage> on it"""
    P = np.load(OUT / "runs" / stage / "poses_merged.npz")["lead"]
    if base == "v43":
        freeze_v43(frozen)
        bname = frozen
    else:
        bname = base
    fb = load_fit(bname)
    info = write_fit_v4(fb, {"lead": P, "follow": hc.full_pose(fb["follow"])}, fit_dir("v43"), note=fit_note("v43", bname), base_dir=FITS[bname])
    cp = curation_file("v43")
    c = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
    c.setdefault("stages", {})[stage] = {"base_tag": bname, "veto": {}}
    c["fit_stages"] = list((c.get("frozen_stages") or {}).get(bname, c.get("fit_stages") or [])) + [stage]
    cp.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    return info


# ------------------------------------------------------------------------------------------------ v43 fix round 2 (2026-10-23, the owner's times)
#
# The owner (2026-10-23): his right hand should usually rest on her upper-left back; the hand holds = the detected poses; only his hand <->
# her shoulder / back collision-checked; "especially around 1:30 and 2:30 and 4:50". After fix 1 (frozen as v43g) 2:30 was worse than the
# detected input, the 1:30 counterbalance arm a "V" hooked over the top of her shoulder, his hand in her neck at 92.6 / 152.5 / 240.5 s.
# Round 2 works only in the owner's ranges (FIX2_RANGES; else v43g exactly):
#   2:30   fix2_mix: per frame the INPUT arm where the input's palm is on her upper-left back / behind her coronal plane and not > 2 cm in her
#          torso (nor > 1 cm in her neck); the input arm with only the hand pushed out of her (fix2_push, the hand's rotation kept) where it
#          is there but too deep; the remaining frames re-posed onto her upper-left back by carrying the hand's pose IN HER SPINE'S FRAME from
#          the nearest such frames (interpolated between two; a resting hand moves with her back), the arm re-solved (arm_keep_hand, the
#          elbow nearest the input's); the composite blended with the base (raised-cosine ramps >= 0.4 s) where the candidate is better;
#   1:30   fix1_window with the counterbalance variants FIX2_CB_VARIANTS (a level, near-straight arm: the sag weighed 5x; the palm at her
#          left back edge) and fix2_check: his forearm against / under her raised left upper arm is natural contact (the owner: only his
#          hand <-> her shoulder / back is checked) - reported, never a veto below FIX2_LUA_VETO_CM;
#   neck   fix2 pushout (her torso frame: down / back only, never onto her front) where a window keeps his hand in her neck;
#   front  (time allowing) the remaining front runs with fix1_window + fix2_check; the kept fix-1 holds (FIX2_PROTECT) never inside a span.
# Output: stage "fix2" of curation_v43.json on v43g, the fit v43 (fix2_finish: the parts joined, the collar bounded again).

FIX2_STAGE = "fix2"
FIX2_BASE = "v43g"
FIX2_RANGES = {"1m30": (80.0, 101.3), "2m30": (140.0, 160.0), "neck240": (240.3, 240.9)}
FIX2_NECK_RANGES = {"neck240": (240.3, 240.9), "neck152": (152.3, 153.1), "neck92": (92.4, 93.0)}   # short neck windows (by --kinds)
FIX2_FRONT_RANGES = {"f54": (53.6, 55.4), "f62": (61.7, 65.6), "f107": (106.4, 115.4), "f322": (321.6, 326.5)}
FIX2_PROTECT = ((28.27, 46.6), (56.8, 61.2), (272.8, 292.8))     # kept / good holds no fix-2 span may touch (seconds)
FIX2_LUA_VETO_CM = 6.0                 # his forearm in her left upper arm: advisory below this (owner: her raised arm rests on his)
FIX2_CB_VARIANTS = ({"w_sag": 300.0, "sag_free_m": 0.02, "upper_arm_allow_m": 0.03, "w_target_pen": 1.0},
                    {"w_sag": 300.0, "sag_free_m": 0.02, "upper_arm_allow_m": 0.03, "w_target_pen": 1.0, "cb_ideal": (20.0, 0.03)},
                    {"w_sag": 300.0, "sag_free_m": 0.02, "upper_arm_allow_m": 0.02, "w_target_pen": 2.0, "cb_ideal": (30.0, 0.06),
                     "w_coco_elbow": 150.0},
                    {})
FIX2_PUSH_DIRS = ((0.0, -1.0, 0.0), (0.0, -0.7, -0.7), (0.0, 0.0, -1.0), (0.7, -0.7, 0.0), (0.5, -0.6, -0.6), (0.7, 0.0, -0.7))
# fix2 pushout's ways for his wrist out of her neck, in her torso frame: down / back / down-left - never toward her front


def fix2_kind(t0: float, t1: float, front: bool = False) -> str | None:
    """the fix-2 range a window (core seconds t0..t1) belongs to, or None"""
    rng = dict(FIX2_RANGES)
    if front:
        rng.update(FIX2_FRONT_RANGES)
    for nm, (a, b) in rng.items():
        if t1 >= a and t0 <= b:
            return nm
    return None


def fix2_clip_span(win: dict, T: int, min_ramp: int = 12) -> dict | None:
    """a window whose span stays out of FIX2_PROTECT (its ramps shortened, the core moved in to keep >= min_ramp frames of ramp), or None"""
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    for a_s, b_s in FIX2_PROTECT:
        a, b = int(round(a_s * HZ)), int(round(b_s * HZ))
        if c0 <= b and c1 >= a:
            return None                                      # the core inside a protected hold
        if s0 <= b < c0:
            s0 = b + 1
        if c1 < a <= s1:
            s1 = a - 1
    if c0 - s0 < min_ramp:
        c0 = s0 + min_ramp
    if s1 - c1 < min_ramp:
        c1 = s1 - min_ramp
    if c1 - c0 < 3:
        return None
    w = dict(win)
    w["core"], w["span"] = [int(c0), int(c1)], [int(max(0, s0)), int(min(T - 1, s1))]
    w["t"] = [round(c0 / HZ, 2), round(c1 / HZ, 2)]
    return w


def fix2_push(ctx: dict, ks: np.ndarray, P: np.ndarray, sel: np.ndarray, kf: dict, VF: np.ndarray, rounds: int = 4, tol_m: float = 0.012,
              margin_m: float = 0.006, max_m: float = 0.08, sigma: float = 2.0) -> tuple[np.ndarray, list]:
    """his hand pushed out of her (her exported skin) on the frames sel of the window ks (poses P (W,55,3)): the wrist moved by the deepest
    hand point's depth + margin along its way out of her torso / collars / upper arms - out of her neck / head: down and back in her torso
    frame (never over her shoulder to her front) - the hand's global rotation kept, the arm re-solved (arm_keep_hand); smoothed in time
    (sigma frames, only the frames within 6 of sel move), <= max_m in all, rounds -> (P, per-round report)"""
    from scipy.ndimage import binary_dilation, gaussian_filter1d
    fl = ctx["fit_lead"]
    P = np.asarray(P, float)
    W = len(ks)
    sel = np.asarray(sel, bool)
    rep = []
    if not sel.any():
        return P.copy(), rep
    o0 = posed(fl, ks, P)
    W0, G21 = o0["J"][:, WRIST["right"]].copy(), o0["R"][:, WRIST["right"]].copy()
    Fr = torso_frame(kf["J"])
    down_back = _unit(np.einsum("kij,j->ki", Fr, np.array([0.0, -0.6, -0.8])))
    hand_ids = ctx["ids"]["hand"][::2]
    ref = ctx["P_in"].copy()
    ref[ks] = P
    dil = binary_dilation(sel, iterations=6)
    shift = np.zeros((W, 3))
    Pc = P.copy()
    for _ in range(rounds):
        o_ = posed(fl, ks, Pc)
        VH = ctx["skinL"].verts(o_["J"], o_["R"], hand_ids)
        dt, ut = hand_way_out(ctx["surfs"], VH, VF, ("torso_arms",))
        dn, _ = hand_way_out(ctx["surfs"], VH, VF, ("neck",))
        dep = np.maximum(dt, dn)
        way = np.where((dn >= dt)[:, None], down_back, ut)
        bad = sel & (dep > tol_m)
        rep.append({"frames": int(bad.sum()), "depth_cm_max": round(float(dep[sel].max()) * 100, 2)})
        if not bad.any():
            break
        need = np.where(bad[:, None], (dep + margin_m)[:, None] * way, 0.0)
        tot = shift + need
        if W > 3:
            tot = gaussian_filter1d(tot, sigma, axis=0, mode="nearest")
            # the smoothing must not shrink a bad frame's push below what it needs
            sc = np.linalg.norm(shift + need, axis=1) / np.maximum(np.linalg.norm(tot, axis=1), 1e-9)
            tot = tot * np.where(bad, np.maximum(sc, 1.0), 1.0)[:, None]
        tot = tot * dil[:, None]
        tot *= np.minimum(1.0, max_m / np.maximum(np.linalg.norm(tot, axis=1), 1e-9))[:, None]
        shift = tot
        mv = np.linalg.norm(shift, axis=1) > 1e-5
        Pn, _ = arm_keep_hand(fl, ref, ref, ks[mv], X_wrist=W0[mv] + shift[mv], G21=G21[mv], twist_cap_deg=88.0)
        Pc = P.copy()
        Pc[mv] = Pn[ks[mv]]
    return Pc, rep


def fix2_mix(ctx: dict, win: dict, cfg: "Fix1Cfg", gap_s: float = 3.0, min_s: float = 0.4, miss_ok_m: float = 0.02,
             sigma_adj: float = 8.0, pull: tuple = (0.05, 0.035), base_anchors: bool = False) -> dict:
    """2:30 (see the section): the INPUT arm; where the input's palm is on her upper-left back / behind her coronal plane but in her (or
    off her) the hand laid flat on her skin THERE: the palm centre moved to the exit of the ray from her spine axis through it (+ lift: its
    angle round her and its height kept), the hand turned by the smallest rotation that makes the palm face her (the input's finger
    direction and wrist twist otherwise kept), the fingers opened, the arm re-solved by arm_keep_hand (the input's elbow hinge / swivel) -
    the adjustment faded in / out over >= 0.4 s (sigma_adj frames) against the frames where the input is fine; pushed out of her where it
    still sinks in (fix2_push); frames whose input palm is on her front carry the hand's pose in her spine's frame from the nearest such
    frames (interpolated between two) where the arm reaches; the composite blended with the base (raised cosines >= min_s) where better
    -> like fix1_window ({"ks", "P", "stats", "VF", "kf"})"""
    from scipy.ndimage import gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    fl = ctx["fit_lead"]
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    ks = np.arange(s0, s1 + 1)
    W = len(ks)
    core = (ks >= c0) & (ks <= c1)
    cls = ctx["cls"][ks]
    hh = ctx["names"][ks] == "hand_hold"
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    Jf = kf["J"]
    VF = ctx["skinF"].verts(Jf, kf["R"])
    Pb = ctx["P_base"][ks].copy()
    Pi = ctx["P_in"][ks].copy()
    tor = core & (cls != "") & ~hh
    arm_j = (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55))
    nmin = int(round(min_s * HZ))
    ref = ctx["P_in"].copy()

    def meas(P_):
        m_ = fix1_metrics(fl, P_, ks, kf, ctx["sites_f"], cls)
        s_ = skin_contact(fl, P_, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
        on_ = m_["ulb"] | (m_["z"] <= 0.0)
        return m_, s_, on_ & (s_["hand"] <= 0.02) & (s_["neck"] <= 0.01), on_
    m_i, s_i, good_i, on_i = meas(Pi)
    good_i &= tor
    loc_ok = tor & on_i & (s_i["neck"] <= 0.02)
    stats = {"frames": int(W), "core": [int(c0), int(c1)], "fallback": "mix", "input_good": int(good_i.sum()), "input_place_ok": int(loc_ok.sum())}
    if not loc_ok.any():
        stats["why"] = "no input frame on her upper-left back / behind her"
        return {"ks": ks, "P": Pb, "stats": stats, "VF": VF, "kf": kf}
    # ---- the input's hand laid flat on her skin at its place (where it is in her / off her)
    oi = posed(fl, ks, Pi)
    Wi, G21i = oi["J"][:, WRIST["right"]], oi["R"][:, WRIST["right"]]
    pci = m_i["pc"]
    tp = fix2_radial_exit(ctx, Jf, VF, pci, cfg.lift_m)
    # ... drawn along her skin toward the nearest upper-left-back site where it lies farther than 6.5 cm from them (a palm low on her back
    # or at her side goes up to her shoulder blade: the owner's place for it)
    spts = np.stack([site_world(ctx["sites_f"][s_], Jf, kf["R"])[0] for s_ in ULB5], 1)
    dd = np.linalg.norm(spts - tp[:, None], axis=2)
    Sn = spts[np.arange(W), dd.argmin(1)]
    dmin = dd.min(1)
    far = dmin > pull[0]
    if far.any():
        tq = np.where(far[:, None], Sn + (tp - Sn) * (pull[1] / np.maximum(dmin, 1e-9))[:, None], tp)
        tq2, _ = fix1_project(ctx["surfs"]["torso_lsh"], tq, VF, cfg.lift_m, Jf, sigma=4.0)
        tp = np.where(far[:, None], tq2, tp)
    stats["pulled_to_ulb"] = int((far & loc_ok & ~good_i).sum())
    _, tn = fix1_project(ctx["surfs"]["torso_lsh"], tp, VF, 0.0, Jf, sigma=4.0)
    adj = (loc_ok & ~good_i).astype(float)
    if W > 3:
        # slow fades (the elbow straightens to reach the laid-flat place: fast fades snapped it by 75 deg in 4 frames)
        adj = np.clip(gaussian_filter1d(adj, sigma_adj, mode="nearest") * 1.4, 0.0, 1.0)
    adj *= ~hh
    Rm = min_rot(hc.palm_normal(oi["J"], "right"), -tn)
    rv = Ro.from_matrix(Rm).as_rotvec()
    if W > 3:
        rv = gaussian_filter1d(rv, 3.0, axis=0, mode="nearest")
    off = np.einsum("kji,kj->ki", G21i, pci - Wi)                      # the palm centre in the wrist's frame (fingers do not move it)
    tps = gaussian_filter1d(tp, 3.0, axis=0, mode="nearest") if W > 3 else tp
    G21n = rotvec_to_mat(rv * adj[:, None]) @ G21i
    Xp = pci + adj[:, None] * (tps - pci)
    Xw = Xp - np.einsum("kij,kj->ki", G21n, off)
    Pa = Pi.copy()
    mv = np.flatnonzero(adj > 1e-3)
    if len(mv):
        Pn, inf_ = arm_keep_hand(fl, ref, ref, ks[mv], X_wrist=Xw[mv], G21=G21n[mv], twist_cap_deg=85.0)
        Pa[mv] = Pn[ks[mv]]
        # where that twists his wrist past 88 deg or the arm cannot reach the place (the leans: she is beyond his straight arm), the
        # adjustment fades out (the input arm there; pushed out of her below) - faded over >= 0.4 s
        tw_ = np.abs(wrist_angles(Pa, "right")["twist"])
        badj = np.zeros(W, bool)
        oo_ = posed(fl, ks[mv], Pa[mv])
        sag_ = elbow_sag(oo_["J"][:, SHOULDER["right"]], oo_["J"][:, ELBOW["right"]], oo_["J"][:, WRIST["right"]])
        fx_ = elbow_angles(Pa, "right")["flexion"]
        # ... or straightens his elbow past 12 deg (the place out of his reach: the elbow snaps straight)
        badj[mv] = (tw_[mv] > 88.0) | (inf_["wrist_miss_m"] > 0.015) | ((cls[mv] == "cb") & (sag_ > 0.09)) | ((fx_[mv] < 12.0) & (fx_[mv] < elbow_angles(Pi, "right")["flexion"][mv] - 15.0))
        if badj.any():
            from scipy.ndimage import binary_dilation as _bd
            keep_ = 1.0 - np.clip(gaussian_filter1d(_bd(badj, iterations=4).astype(float), 3.0, mode="nearest") * 1.5, 0.0, 1.0)
            adj = adj * keep_
            G21n = rotvec_to_mat(rv * adj[:, None]) @ G21i
            Xp = pci + adj[:, None] * (tps - pci)
            Xw = Xp - np.einsum("kij,kj->ki", G21n, off)
            Pa = Pi.copy()
            mv = np.flatnonzero(adj > 1e-3)
            if len(mv):
                Pn, inf_ = arm_keep_hand(fl, ref, ref, ks[mv], X_wrist=Xw[mv], G21=G21n[mv], twist_cap_deg=85.0)
                Pa[mv] = Pn[ks[mv]]
            stats["adj_dropped"] = int(badj.sum())
        flex_o, fixed_o = hc.finger_template(fl["betas"], "right", "open")
        fo = (flex_o + fixed_o)
        for j in range(15):
            Pa[mv, 40 + j] = _slerp_rotvec(Pi[mv, 40 + j], np.tile(fo[j], (len(mv), 1)), adj[mv])
    m_a, s_a, _, _ = meas(Pa)
    dp = loc_ok & ((s_a["hand"] > 0.012) | (s_a["neck"] > 0.004))
    if dp.any():
        Pa, stats["push_anchor"] = fix2_push(ctx, ks, Pa, dp, kf, VF)
    m_a, s_a, good_a, _ = meas(Pa)
    anchor = loc_ok & good_a
    if base_anchors:
        # also the base's frames outside the core whose palm rests on her upper-left back (a kept neighbour window's hand): the core
        # carries that place on (her hand stays where it was on her back)
        m_b0, s_b0, _, _ = meas(Pb)
        ba = ~core & ~hh & (cls != "") & m_b0["ulb"] & (s_b0["hand"] <= 0.02) & (s_b0["neck"] <= 0.005)
        Pa[ba] = Pb[ba]
        m_a, s_a, good_a, _ = meas(Pa)
        anchor = anchor | ba
        stats["base_anchors"] = int(ba.sum())
    stats["anchors"] = int(anchor.sum())
    stats["anchor_ulb"] = int((anchor & m_a["ulb"]).sum())
    # ---- the other core frames: the hand's pose carried in her spine-3 frame from the anchors
    oa = posed(fl, ks, Pa)
    Rsp, Osp = kf["R"][:, SPINE3], Jf[:, SPINE3]
    xa = np.einsum("kji,kj->ki", Rsp, oa["J"][:, WRIST["right"]] - Osp)
    qa = Ro.from_matrix(np.transpose(Rsp, (0, 2, 1)) @ oa["R"][:, WRIST["right"]]).as_rotvec()
    # carried from the anchors on her upper-left back where there are such (the owner: his hand on her upper-left back), else any anchor
    au = anchor & m_a["ulb"]
    ai = np.flatnonzero(au if au.sum() >= max(3, 0.2 * anchor.sum()) else anchor)
    fill = ~anchor & ~hh & core & ~good_i
    Pc = Pa.copy()
    miss = np.zeros(W)
    if len(ai) and fill.any():
        G = int(round(gap_s * HZ))
        xf, qf, fing = xa.copy(), qa.copy(), Pa[:, 40:55].copy()
        for i in np.flatnonzero(fill):
            p_, q_ = ai[ai < i], ai[ai > i]
            p, q = (int(p_[-1]) if len(p_) else None), (int(q_[0]) if len(q_) else None)
            if p is not None and q is not None and q - p <= G:
                t = (i - p) / (q - p)
                t = t * t * (3 - 2 * t)
                xf[i] = (1 - t) * xa[p] + t * xa[q]
                qf[i] = _slerp_rotvec(qa[p][None], qa[q][None], np.array([t]))[0]
                fing[i] = np.stack([_slerp_rotvec(Pa[p, 40 + j][None], Pa[q, 40 + j][None], np.array([t]))[0] for j in range(15)])
            else:
                n_ = p if (q is None or (p is not None and i - p <= q - i)) else q
                xf[i], qf[i], fing[i] = xa[n_], qa[n_], Pa[n_, 40:55]
        fi_ = np.flatnonzero(fill)
        Xw2 = Osp[fi_] + np.einsum("kij,kj->ki", Rsp[fi_], xf[fi_])
        G21c = Rsp[fi_] @ rotvec_to_mat(qf[fi_])
        Pn, info = arm_keep_hand(fl, ref, ref, ks[fi_], X_wrist=Xw2, G21=G21c, twist_cap_deg=85.0)
        Pc[fi_] = Pn[ks[fi_]]
        Pc[fi_, 40:55] = fing[fi_]
        miss[fi_] = info["wrist_miss_m"]
        m_c, s_c, _, _ = meas(Pc)
        dp = fill & (miss <= miss_ok_m) & ((s_c["hand"] > 0.012) | (s_c["neck"] > 0.004))
        if dp.any():
            Pc2, stats["push_carry"] = fix2_push(ctx, ks, Pc, dp, kf, VF)
            Pc[fill] = Pc2[fill]
        if W > 5:
            Ps = smooth_rotations(Pc, (SHOULDER["right"], ELBOW["right"], WRIST["right"]), 1.5)
            Pc[fill] = Ps[fill]
    m_c, s_c, good_c, on_c = meas(Pc)
    m_b, s_b, good_b, _ = meas(Pb)
    bad_c = fix1_frame_badness(m_c, s_c, cls)
    bad_b = fix1_frame_badness(m_b, s_b, cls)
    fxc = elbow_angles(Pc, "right")["flexion"]
    fxi = elbow_angles(Pi, "right")["flexion"]
    pref = tor & (miss <= miss_ok_m) & (good_c | (bad_c < bad_b - 0.2)) & ~((fxc < 12.0) & (fxc < fxi - 15.0))
    for a, b in _runs(~pref, 0):
        if 0 < a and b < W - 1 and (b - a + 1) < nmin and not hh[a:b + 1].any() and (miss[a:b + 1] <= 0.04).all():
            pref[a:b + 1] = True
    for a, b in _runs(pref, 0):
        if (b - a + 1) < nmin:
            pref[a:b + 1] = False
    D = np.linalg.norm(m_c["pc"] - m_b["pc"], axis=1)
    w = pref.astype(float)
    for a, b in _runs(pref, 0):
        for e, sgn in ((a, -1), (b, 1)):
            L = int(np.clip(np.ceil(max(D[e], 0.01) / (cfg.blend_mps / HZ) * np.pi / 2), nmin, 60))
            for i in range(1, L + 1):
                k = e + sgn * i
                if k < 0 or k >= W or hh[k] or miss[k] > 0.04:
                    break
                w[k] = max(w[k], 0.5 + 0.5 * np.cos(np.pi * i / (L + 1)))
    P = Pb.copy()
    on = w > 0
    for j in arm_j:
        P[on, j] = _slerp_rotvec(Pb[on, j], Pc[on, j], w[on])
    P[w >= 1.0] = Pc[w >= 1.0]
    # the speed rules inside the window (the flagged steps low-passed where the composite left the base), then his hand out of her
    if ctx.get("p99") is not None and on.any():
        P, stats["speed_repair"] = speed_repair(ctx, P, rounds=8, half=6, sigma=3.0, ks=ks, allowed=on)
        # elbow snaps (its flexion faster than max(the input's p99, the input there) + 90 deg/s: the arm straightening to reach a carried /
        # laid-flat place): the arm (not the collar) low-passed round them, more strongly, until none is left (<= 6 rounds)
        from scipy.ndimage import binary_dilation as _bd2
        fli_ = elbow_angles(Pi, "right")["flexion"]
        for _ in range(6):
            fl_ = elbow_angles(P, "right")["flexion"]
            snap = np.abs(np.diff(fl_)) * HZ > np.maximum(ctx["p99"]["flex"], np.abs(np.diff(fli_)) * HZ) + 90.0
            snap &= on[1:] | on[:-1]
            if not snap.any():
                break
            m_ = np.zeros(W, bool)
            m_[1:] |= snap
            m_[:-1] |= snap
            m_ = _bd2(m_, iterations=8) & on
            Ps_ = smooth_rotations(P, (SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)), 4.0)
            P[m_] = Ps_[m_]
            stats["snap_repair_frames"] = stats.get("snap_repair_frames", 0) + int(m_.sum())
    m_f, s_f, _, _ = meas(P)
    dpf = core & ~hh & ((s_f["hand"] > 0.015) | (s_f["neck"] > 0.005))
    if dpf.any():
        P, stats["push_final"] = fix2_push(ctx, ks, P, dpf, kf, VF)
    stats.update({"carried": int(fill.sum()), "carry_miss_cm_p90": round(float(np.percentile(miss[fill], 90)) * 100, 2) if fill.any() else 0.0,
                  "preferred": int(pref.sum()), "blended": int(((w > 0) & (w < 0.999)).sum()), "cand_good_core": int((good_c & tor).sum()),
                  "base_good_core": int((good_b & tor).sum()), "cand_ulb_core": int((m_c["ulb"] & tor).sum()),
                  "input_arm_frames": int(((np.abs(P - Pi).max((1, 2)) == 0) & tor).sum())})
    return {"ks": ks, "P": P, "stats": stats, "VF": VF, "kf": kf, "w": w}


def fix2_radial_exit(ctx: dict, Jf: np.ndarray, VF: np.ndarray, X: np.ndarray, lift_m: float) -> np.ndarray:
    """(W,3) per frame the exit of the ray from her spine axis (spine 3 + the point's height along her spine, spine_frame) through the point X
    out of her torso + collars (her exported skin VF), lifted lift_m along the face normal; where no exit: her closest skin point (+ lift)"""
    Fr = spine_frame(Jf)
    O3 = Jf[:, SPINE3]
    y = np.einsum("ki,ki->k", Fr[:, :, 1], X - O3)
    O = O3 + y[:, None] * Fr[:, :, 1]
    D = _unit(X - O, Fr[:, :, 2] * -1.0)
    faces = ctx["surfs"]["faces_torso"]
    t, fi = hc.ray_first_hit(O, D, VF, faces, ctx["device"])
    hit = np.isfinite(t) & (fi >= 0) & (t < 0.40)
    out, _ = fix1_project(ctx["surfs"]["torso_lsh"], X, VF, lift_m, Jf, sigma=5.0)
    if hit.any():
        n = face_normals_np(np.asarray(VF[hit], float), faces, fi[hit])
        n = np.where(((n * D[hit]).sum(1) < 0)[:, None], -n, n)
        out[hit] = O[hit] + t[hit, None] * D[hit] + lift_m * n
    return out


def fix2_down(ctx: dict, win: dict, cfg: "Fix1Cfg", site: str = "shoulder_blade_L") -> dict:
    """a hand at the back of her neck / head (verify F7: at 152.6 / 240.6 s his palm sits 21-24 cm up her spine, at her neck): the palm moved
    DOWN her back onto her left shoulder blade (the ULB site, laid on her exported skin + lift), the hand turned by the smallest rotation that
    makes the palm face her, the fingers opened, the arm re-solved (arm_keep_hand) - fully on the core frames whose hand is in her neck
    (+-6 frames), faded in / out by raised cosines (>= 0.4 s) inside the span; then pushed out of her where it sinks in -> like fix1_window"""
    from scipy.ndimage import binary_dilation
    fl = ctx["fit_lead"]
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    ks = np.arange(s0, s1 + 1)
    W = len(ks)
    core = (ks >= c0) & (ks <= c1)
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    Jf = kf["J"]
    VF = ctx["skinF"].verts(Jf, kf["R"])
    Pb = ctx["P_base"][ks].copy()
    hh = ctx["names"][ks] == "hand_hold"
    sc = skin_contact(fl, Pb, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    on = binary_dilation(core & (sc["neck"] > 0.004), iterations=6) & ~hh
    stats = {"frames": int(W), "core": [int(c0), int(c1)], "fallback": "neck", "neck_frames": int((core & (sc["neck"] > 0.004)).sum())}
    if not on.any():
        return {"ks": ks, "P": Pb, "stats": stats, "VF": VF, "kf": kf}
    w = on.astype(float)
    L = 30
    for a, b in _runs(on, 0):
        for e, sgn in ((a, -1), (b, 1)):
            for i in range(1, L + 1):
                k = e + sgn * i
                if k < 0 or k >= W or hh[k]:
                    break
                w[k] = max(w[k], 0.5 + 0.5 * np.cos(np.pi * i / (L + 1)))
    ob = posed(fl, ks, Pb)
    Wb, G21b = ob["J"][:, WRIST["right"]], ob["R"][:, WRIST["right"]]
    pcb = hc.palm_centre(ob["J"], "right")
    X0 = site_world(ctx["sites_f"][site], Jf, kf["R"])[0]
    X, n = fix1_project(ctx["surfs"]["torso_lsh"], X0, VF, cfg.lift_m, Jf, sigma=3.0)
    Rm = min_rot(hc.palm_normal(ob["J"], "right"), -n)
    from scipy.spatial.transform import Rotation as Ro
    rv = Ro.from_matrix(Rm).as_rotvec()
    G21n = rotvec_to_mat(rv * w[:, None]) @ G21b
    off = np.einsum("kji,kj->ki", G21b, pcb - Wb)
    # the way down her back follows her skin (+ lift): the straight line between the neck and her shoulder blade cuts into her shoulder
    Xl = pcb + w[:, None] * (X - pcb)
    Xs_, _ = fix1_project(ctx["surfs"]["torso_lsh"], Xl, VF, cfg.lift_m, Jf, sigma=2.0)
    sw = np.clip(4.0 * w, 0.0, 1.0)[:, None]
    Xp = Xl + sw * (Xs_ - Xl)
    Xw = Xp - np.einsum("kij,kj->ki", G21n, off)
    ref = ctx["P_in"].copy()
    ref[ks] = Pb
    mv = np.flatnonzero(w > 1e-3)
    P = Pb.copy()
    Pn, info = arm_keep_hand(fl, ref, ref, ks[mv], X_wrist=Xw[mv], G21=G21n[mv], twist_cap_deg=88.0)
    P[mv] = Pn[ks[mv]]
    flex_o, fixed_o = hc.finger_template(fl["betas"], "right", "open")
    fo = flex_o + fixed_o
    for j in range(15):
        P[mv, 40 + j] = _slerp_rotvec(Pb[mv, 40 + j], np.tile(fo[j], (len(mv), 1)), w[mv])
    sc1 = skin_contact(fl, P, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    dp = (w > 0) & ((sc1["hand"] > 0.012) | (sc1["neck"] > 0.004))
    if dp.any():
        P, stats["push"] = fix2_push(ctx, ks, P, dp, kf, VF)
    stats.update({"moved_frames": int(len(mv)), "wrist_miss_cm_max": round(float(info["wrist_miss_m"].max()) * 100, 2),
                  "move_cm_max": round(float(np.linalg.norm(X - pcb, axis=1)[on].max()) * 100, 1)})
    return {"ks": ks, "P": P, "stats": stats, "VF": VF, "kf": kf}


def fix2_neck(ctx: dict, win: dict, cfg: "Fix1Cfg") -> dict:
    """the base with only his hand moved out of her neck / head (fix2_push: down and back in her torso frame, out of her torso along the
    way out), on the core frames whose hand is in her neck (+-4 frames), then the speed rules where it moved -> like fix1_window"""
    from scipy.ndimage import binary_dilation
    c0, c1 = win["core"]
    s0, s1 = win["span"]
    ks = np.arange(s0, s1 + 1)
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    VF = ctx["skinF"].verts(kf["J"], kf["R"])
    Pb = ctx["P_base"][ks].copy()
    core = (ks >= c0) & (ks <= c1)
    sc = skin_contact(ctx["fit_lead"], Pb, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    sel = binary_dilation(core & (sc["neck"] > 0.004), iterations=4)
    P, rep = fix2_push(ctx, ks, Pb, sel, kf, VF, rounds=6, tol_m=0.004, margin_m=0.008)
    mvd = np.abs(P - Pb).max((1, 2)) > 0
    sp = None
    if ctx.get("p99") is not None and mvd.any():
        P2, sp = speed_repair(ctx, P, rounds=8, half=6, sigma=3.0, ks=ks, allowed=binary_dilation(mvd, iterations=6))
        sc2 = skin_contact(ctx["fit_lead"], P2, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
        sc1 = skin_contact(ctx["fit_lead"], P, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
        back = binary_dilation(sc2["neck"] > np.maximum(sc1["neck"], 0.006), iterations=6)
        P2[back] = P[back]
        P = P2
    return {"ks": ks, "P": P, "stats": {"frames": int(len(ks)), "core": [int(c0), int(c1)], "fallback": "neck", "push": rep, "speed_repair": sp,
                                        "frames_moved": int(mvd.sum())}, "VF": VF, "kf": kf}


def fix2_pushout(ctx: dict, win: dict, cfg: "Fix1Cfg") -> dict:
    """fix1_pushout with fix 2's ways (FIX2_PUSH_DIRS: down / back in her torso frame, never toward her front)"""
    global FIX1_PUSH_DIRS
    keep = FIX1_PUSH_DIRS
    try:
        FIX1_PUSH_DIRS = FIX2_PUSH_DIRS
        return fix1_pushout(ctx, win, cfg)
    finally:
        FIX1_PUSH_DIRS = keep


def fix2_check(ctx: dict, r: dict, cfg: "Fix1Cfg", kind: str) -> dict:
    """fix1_check with the owner's rule (2026-10-23): his forearm against / under her raised left upper arm is natural contact - its
    "forearm into her left upper arm" veto only above FIX2_LUA_VETO_CM (reported in "notes"); in the counterbalance leans (1:30) a small
    loss of upper-left-back share outside the leans is accepted when the sagging elbow is fixed and the front share does not grow; the wrist
    twist rule +8 % of the core; plus: the share of core frames behind her coronal plane (z <= 0) must not fall, and at 2:30 the ULB share
    must reach the input's"""
    chk = fix1_check(ctx, r, cfg)
    num = chk["numbers"]
    b, n, i_ = num["base"], num["new"], num["input"]
    ks = r["ks"]
    c0, c1 = r["stats"]["core"]
    core = (ks >= c0) & (ks <= c1)
    nc = int(core.sum())
    cls = ctx["cls"][ks]
    notes = list(num.get("notes") or [])
    why = []
    def fore_t():
        # frames whose forearm is > 4 cm in her where its deepest point is not in her left upper arm (her torso / collars / right arm)
        out_ = {}
        for nm_, P_ in (("input", ctx["P_in"][ks]), ("base", ctx["P_base"][ks]), ("new", r["P"])):
            sc_ = skin_contact(ctx["fit_lead"], P_, ks, r["kf"], r["VF"], ctx["skinL"], ctx["ids"], ctx["surfs"])
            out_[nm_] = int(((sc_["fore"] > 0.04) & (sc_["fore_lua"] < sc_["fore"] - 0.005)).sum())
        num["fore_torso_gt4"] = out_
        return out_
    Wi_, Ei_ = wrist_angles(ctx["P_in"][ks], "right"), elbow_angles(ctx["P_in"][ks], "right")
    wi_in = {"bend_gt100": int((Wi_["bend"] > 100).sum()), "twist_gt90": int(((np.abs(Wi_["twist"]) > 90) & (Wi_["bend"] < 60)).sum()),
             "hyper": int((Ei_["flexion"] < -5).sum()), "off_hinge": int((np.abs(Ei_["off_hinge"]) > 35).sum())}
    num["input"].update(wi_in)
    for w_ in chk["why"]:
        if w_.startswith("forearm into her left upper arm") and n["fore_lua_max_cm"] <= FIX2_LUA_VETO_CM:
            notes.append("advisory (owner: his forearm under her raised left arm is natural contact): " + w_)
            continue
        if w_.startswith("forearm into her > 4 cm"):
            ft = fore_t()
            lim_ = max(5, int(0.06 * nc))
            if ft["new"] <= max(ft["base"], ft["input"]) + lim_:
                notes.append(f"advisory (the deep forearm point in her raised left upper arm; elsewhere in her > 4 cm {ft['input']} (input) / "
                             f"{ft['base']} -> {ft['new']}): " + w_)
                continue
        if w_.startswith("twist_gt90") and n["twist_gt90"] <= max(b["twist_gt90"], wi_in["twist_gt90"]) + max(5, int(0.08 * nc)):
            notes.append("advisory: " + w_)
            continue
        if any(w_.startswith(k_) for k_ in ("bend_gt100", "hyper", "off_hinge")):
            k_ = w_.split(" ")[0]
            if n[k_] <= max(b[k_], wi_in[k_]) + max(3, int(0.03 * nc)):
                notes.append("advisory (the input's own): " + w_)
                continue
        if (w_.startswith("location worse") and kind == "1m30" and n["front"] <= b["front"] + 0.02 and n["cb_sag_gt10"] < b["cb_sag_gt10"]
                and (n.get("ulb_not_cb") or 0) >= (b.get("ulb_not_cb") or 0) - 0.10):
            notes.append("advisory (the lean's arm straightened): " + w_)
            continue
        if (w_.startswith("location worse") and kind == "2m30" and n["front"] <= b["front"] + 0.04
                and (n.get("ulb_not_cb") or 0) >= (b.get("ulb_not_cb") or 0) + 0.2):
            notes.append("advisory (2:30: onto her upper-left back, the front share +<= 0.04): " + w_)
            continue
        why.append(w_)
    tor = core & (cls != "")
    zs = {}
    for nm, P in (("input", ctx["P_in"][ks]), ("base", ctx["P_base"][ks]), ("new", r["P"])):
        m = fix1_metrics(ctx["fit_lead"], P, ks, r["kf"], ctx["sites_f"], cls)
        zs[nm] = round(float((m["z"][tor] <= 0.0).mean()), 3) if tor.any() else None
        num[nm]["z_le0"] = zs[nm]
    if zs["new"] is not None and zs["new"] < zs["base"] - 0.02:
        why.append(f"behind her coronal plane (z <= 0) {zs['base']} -> {zs['new']}")
    gain = kind == "2m30" and n["ulb"] >= b["ulb"] + 0.2 and n["front"] <= b["front"] + 0.02 and n["neck_gt1"] <= b["neck_gt1"]
    if kind == "2m30" and ((n["hand_gt2"] <= 0.5 * b["hand_gt2"] and n["neck_gt1"] <= b["neck_gt1"]) or gain):
        # the hand out of her (half the frames > 2 cm or fewer) or onto her upper-left back (ULB +0.2, the owner's 2:30 request; the
        # input there is far deeper in her): the hand turns over / comes out on the way - the speed / twist rules +10 % of the core, his
        # hand > 5 cm in her + max(4, 3 %) while below the input's, the lean's sag + 3 frames (a documented trade, notes_H2)
        keep_w = []
        for w_ in why:
            k_ = w_.replace("speed rule ", "").split(" ")[0]
            if k_ in ("addfast", "flexfast", "twist_gt90", "fast") and n.get(k_, 0) <= max(b.get(k_, 0), wi_in.get(k_, 0) or 0) + max(6, int(0.10 * nc))                     and (k_ != "fast" or n[k_] <= b[k_] + 6):
                notes.append("advisory (2:30, the hand out of her / onto her upper-left back): " + w_)
                continue
            if gain and w_.startswith("hand into her > 2 / 5 cm") and n["hand_gt5"] <= min(b["hand_gt5"] + max(4, int(0.03 * nc)), i_["hand_gt5"])                     and n["hand_gt2"] <= max(b["hand_gt2"] + max(10, int(0.08 * nc)), 0) and n["hand_gt2"] <= i_["hand_gt2"]:
                notes.append("advisory (2:30 onto her upper-left back, below the input's depth): " + w_)
                continue
            if gain and w_.startswith("counterbalance elbow sag") and n["cb_sag_gt10"] <= b["cb_sag_gt10"] + 3:
                notes.append("advisory (2:30): " + w_)
                continue
            keep_w.append(w_)
        why = keep_w
    # 2:30: per solve window the ULB share must not fall below the base's (the owner window 140-160 s as a whole must reach the input's:
    # checked on the merged fit, validate tables)
    if kind == "2m30" and n["ulb"] < b["ulb"] - 1e-9:
        why.append(f"2:30 ULB {i_['ulb']} (input) / {b['ulb']} (base) -> {n['ulb']}: below the base's")
    num["notes"] = notes
    return {"ok": not why, "why": why, "numbers": num}


def fix2_windows(ctx: dict, cfg: "Fix1Cfg", front: bool = False) -> list[dict]:
    """the fix-2 windows: fix1_plan's windows on the base (with his hand's neck depth) whose core meets a FIX2_RANGES range (front: also the
    FIX2_FRONT_RANGES), clipped clear of FIX2_PROTECT; the neck run at 240.5 s as its own short window"""
    T = len(ctx["cls"])
    torso = np.flatnonzero(ctx["cls"] != "")
    neck = neck_depth_all(ctx, ctx["P_base"], torso)
    ctx["neck_base"] = neck
    out = []
    # 2:30: the whole torso-hold runs of 140-160 s (fix1_plan's chunks and its 3-s ramps left 153.9-156.9 s, the input's hand in front of
    # her, out): per run one window, the ramps >= 14 frames inside the run where a hand hold of his right hand bounds it
    hh = ctx["names"] == "hand_hold"
    has = (ctx["cls"] != "") & ~hh
    a230, b230 = (int(round(x * HZ)) for x in FIX2_RANGES["2m30"])
    for r0, r1 in _runs(has, 0):
        if r1 < a230 or r0 > b230 or (r1 - r0) < 30:
            continue
        r1 = min(r1, b230 + 60)
        lo = r0 - 1
        while lo > max(0, r0 - 90) and not hh[lo - 1]:
            lo -= 1
        hi = r1 + 1
        while hi < min(T - 1, r1 + 90) and not hh[hi + 1]:
            hi += 1
        s0, s1 = lo + 3 if lo < r0 - 3 and hh[lo - 1] else lo, hi - 3 if hh[min(T - 1, hi + 1)] else hi
        c0, c1 = max(r0, s0 + 14), min(r1, s1 - 14)
        if c1 - c0 < 15:
            continue
        nch = int(np.ceil((c1 - c0 + 1) / (6.5 * HZ)))
        edges = np.linspace(c0, c1 + 1, nch + 1).round().astype(int)
        for ci in range(nch):
            e0, e1 = int(edges[ci]), int(edges[ci + 1]) - 1
            sp0 = s0 if ci == 0 else max(s0, e0 - 45)
            sp1 = s1 if ci == nch - 1 else min(s1, e1 + 45)
            out.append({"core": [e0, e1], "span": [int(sp0), int(sp1)], "t": [round(e0 / HZ, 2), round(e1 / HZ, 2)], "kind": "2m30",
                        "chunk": [ci, nch], "classes": {c_: int((ctx["cls"][e0:e1 + 1] == c_).sum()) for c_ in ("ulb", "cb", "back")}})
    for w in fix1_plan(ctx, cfg, neck):
        kind = fix2_kind(*w["t"], front=front)
        if kind is None or kind in ("neck240", "2m30"):
            continue
        w2 = fix2_clip_span(w, T)
        if w2 is None:
            continue
        w2["kind"] = kind
        out.append(w2)
    # the neck runs (240.5 s; 152.6 / 92.7 s when a later continuation asks): a short window round each (its neighbours stay)
    for nm, (a_s, b_s) in FIX2_NECK_RANGES.items():
        a, b = int(round(a_s * HZ)), int(round(b_s * HZ))
        nk = np.flatnonzero(neck[a:b + 1] > 0.005)
        if len(nk):
            k0, k1 = a + int(nk[0]) - 6, a + int(nk[-1]) + 6
            out.append({"core": [k0, k1], "span": [k0 - 24, k1 + 24], "t": [round(k0 / HZ, 2), round(k1 / HZ, 2)], "kind": nm,
                        "chunk": [0, 1], "bad_frames": int(len(nk))})
    out.sort(key=lambda w: w["core"][0])
    return out


def fix2_pass(base: str = FIX2_BASE, part: tuple | None = None, stage: str = FIX2_STAGE, device: str = "cuda", front: bool = True,
              only: list | None = None, kinds: tuple | None = None, out_part: int | None = None, start: str | None = None) -> dict:
    """fix round 2 (see the section) on the base (v43g) -> runs/<stage>/poses_part<i>.npz + report_part<i>.json (part (i, n): the window
    groups g % n == i), or (part None) poses_part0 / report_part0 of all windows; fix2_merge joins and writes"""
    cfg = Fix1Cfg(device=device)
    t0 = time.time()
    ctx = fix1_context(base, device, collar=False)
    ctx["p99"] = speed_refs(ctx["fit_in"])
    if start:
        # a continuation: the windows solved on another part's result (runs/<stage>/poses_part<start>.npz: its kept windows in place)
        ctx["P_base"] = np.load(OUT / "runs" / stage / f"poses_part{start}.npz")["lead"].copy()
    wins = fix2_windows(ctx, cfg, front=front)
    if kinds:
        wins = [w for w in wins if w["kind"] in kinds]
    groups = fix1_groups(wins)
    log(f"fix2: {len(wins)} windows in {max(groups) + 1 if groups else 0} groups ({time.time() - t0:.0f} s): "
        + ", ".join(f"{w['kind']} {w['t']}" for w in wins))
    rep = {"what": "v43 fix round 2 (zouk_contacts.fix2_pass)", "base": base, "part": list(part) if part else None, "start": start, "windows": [],
           "p99_input": ctx["p99"]}
    for wi, win in enumerate(wins):
        if part and groups[wi] % int(part[1]) != int(part[0]):
            continue
        if only and not any(win["t"][0] - 1 <= float(t) <= win["t"][1] + 1 for t in only):
            continue
        t1 = time.time()
        kind = win["kind"]
        tries = []
        if kind == "2m30":
            tries.append(("mix", lambda c_: fix2_mix(ctx, win, c_), cfg))
            # a gentler draw toward the upper-left-back sites (less hand travel: the speed rules)
            tries.append(("mix_soft", lambda c_: fix2_mix(ctx, win, c_, pull=(0.065, 0.055)), cfg))
            tries.append(("mix_carry", lambda c_: fix2_mix(ctx, win, c_, base_anchors=True), cfg))
        isneck = kind.startswith("neck")
        if isneck:
            tries.append(("down", lambda c_: fix2_down(ctx, win, c_), cfg))
            tries.append(("neckpush", lambda c_: fix2_neck(ctx, win, c_), cfg))
            tries.append(("neck2", lambda c_: fix2_pushout(ctx, win, c_), cfg))
        variants = FIX2_CB_VARIANTS if kind == "1m30" else FIX1_VARIANTS
        if not isneck:
            for vi, var in enumerate(variants):
                tries.append((vi, (lambda c_: fix1_window(ctx, win, c_)), Fix1Cfg(**{**asdict(cfg), **var})))
        tried, chk, r, vname = [], None, None, None
        for vname, fn, cfg_v in tries:
            r = fn(cfg_v)
            chk = fix2_check(ctx, r, cfg_v, kind)
            tried.append({"variant": vname, "why": chk["why"], "notes": chk["numbers"].get("notes"),
                          "front_ulb_zle0": [chk["numbers"]["new"]["front"], chk["numbers"]["new"]["ulb"], chk["numbers"]["new"].get("z_le0")],
                          "stats": {k_: v_ for k_, v_ in r["stats"].items() if not isinstance(v_, (list, dict))}})
            if chk["ok"]:
                break
        if not chk["ok"] and chk["numbers"]["base"]["neck_gt1"] > 0 and not isneck:
            r = fix2_neck(ctx, win, cfg)
            chk = fix2_check(ctx, r, cfg, kind)
            vname = "neckpush"
            tried.append({"variant": vname, "why": chk["why"], "notes": chk["numbers"].get("notes"),
                          "front_ulb_zle0": [chk["numbers"]["new"]["front"], chk["numbers"]["new"]["ulb"], chk["numbers"]["new"].get("z_le0")]})
        if not chk["ok"] and chk["numbers"]["base"]["neck_gt1"] > 0 and not isneck:
            r = fix2_pushout(ctx, win, cfg)
            chk = fix2_check(ctx, r, cfg, kind)
            vname = "neck2"
            tried.append({"variant": vname, "why": chk["why"], "notes": chk["numbers"].get("notes"),
                          "front_ulb_zle0": [chk["numbers"]["new"]["front"], chk["numbers"]["new"]["ulb"], chk["numbers"]["new"].get("z_le0")]})
        res = {"window": wi, "group": groups[wi], "kind": kind, "t": win["t"], "core": win["core"], "span": win["span"],
               "adopted": bool(chk["ok"]), "variant": vname, "why": chk["why"], "check": chk["numbers"], "tried": tried,
               "secs": round(time.time() - t1, 1)}
        if chk["ok"]:
            ctx["P_base"][r["ks"]] = r["P"]
        n_ = chk["numbers"]
        log(f"fix2 window {wi + 1}/{len(wins)} {kind} {win['t']}: {'KEPT (' + str(vname) + ')' if chk['ok'] else 'vetoed'} front "
            f"{n_['base']['front']} -> {n_['new']['front']}, ULB {n_['input']['ulb']} (input) / {n_['base']['ulb']} -> {n_['new']['ulb']}, "
            f"sag>10 {n_['base']['cb_sag_gt10']} -> {n_['new']['cb_sag_gt10']}, neck>1 {n_['base']['neck_gt1']} -> {n_['new']['neck_gt1']}"
            f"{'' if chk['ok'] else ' ' + str([t_['why'] for t_ in tried])} ({res['secs']} s)")
        rep["windows"].append(res)
    d = OUT / "runs" / stage
    d.mkdir(parents=True, exist_ok=True)
    i_part = int(out_part) if out_part is not None else (int(part[0]) if part else 0)
    np.savez_compressed(d / f"poses_part{i_part}.npz", lead=ctx["P_base"])
    rep["seconds"] = round(time.time() - t0, 1)
    write_json(d / f"report_part{i_part}.json", py_json(rep))
    return rep


FIX2_WHIPS_S = ((73.3, 73.7), (165.4, 165.8))      # the one-frame fingertip whips fix 1 left (verify: 1.9 / 2.2 m/s)


def fix2_merge(n, base: str = FIX2_BASE, stage: str = FIX2_STAGE, write: bool = True, device: str = "cuda", whips: bool = True) -> dict:
    """the parts of fix2_pass joined on the base (each part changed only its own window groups' frames), the two fingertip whips low-passed
    (FIX2_WHIPS_S, a depth guard), the collar bounded again -> runs/<stage>/ (poses_merged.npz, refine.json, report.json); write: the fit
    v43 = the base (v43g) with these leader poses, curation_v43.json stage <stage> (fit_stages = the base's + [stage])"""
    t0 = time.time()
    ctx = fix1_context(base, device, collar=False)
    ctx["p99"] = speed_refs(ctx["fit_in"])
    d = OUT / "runs" / stage
    P0 = ctx["P_base"].copy()
    P = P0.copy()
    owner = np.full(len(P0), -1)
    rep = {"what": "v43 fix round 2 (zouk_contacts.fix2_pass parts joined by fix2_merge)", "base": base, "parts": n, "windows": []}
    # parts in order, a later part wins on the frames it changed (a continuation part, fix2 --start j, holds part j's windows + its own)
    order = list(range(n)) if isinstance(n, int) else [int(x) for x in n]
    rep["parts"] = order
    seen = set()
    for i in order:
        Pi = np.load(d / f"poses_part{i}.npz")["lead"]
        ch = np.abs(Pi - P0).max((1, 2)) > 0
        clash = ch & (owner >= 0) & (np.abs(Pi - P).max((1, 2)) > 0)
        if clash.any():
            log(f"fix2_merge: part {i} re-sets {int(clash.sum())} frames part(s) {sorted(set(owner[clash].tolist()))} set (a continuation wins)")
        owner[ch] = i
        P[ch] = Pi[ch]
        ri = json.loads((d / f"report_part{i}.json").read_text(encoding="utf-8"))
        for w in ri["windows"]:
            w["part"] = i
            key = (w["kind"], tuple(w["core"]))
            if key in seen:
                # the same window solved again in a continuation: the later solve replaces the earlier entry
                rep["windows"] = [x for x in rep["windows"] if (x["kind"], tuple(x["core"])) != key]
            seen.add(key)
            rep["windows"].append(w)
    rep["windows"] = sorted(rep["windows"], key=lambda w: w["core"][0])
    for i, w in enumerate(rep["windows"]):
        w["window"] = i
    kept = [w for w in rep["windows"] if w["adopted"]]
    extra = []
    if whips:
        # the one-frame fingertip whips: the arm's rotations (not the collar) low-passed on +-8 frames round each (frames already off the
        # input only), put back where that puts his hand deeper into her
        allowed = np.zeros(len(P), bool)
        for a_s, b_s in FIX2_WHIPS_S:
            allowed[int(round(a_s * HZ)) - 8:int(round(b_s * HZ)) + 9] = True
        allowed &= np.abs(P - ctx["P_in"]).max((1, 2)) > 0
        P2, sp = speed_repair(ctx, P, rounds=6, half=4, sigma=2.0, allowed=allowed,
                              joints=(SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)))
        ch = np.flatnonzero(np.abs(P2 - P).max((1, 2)) > 0)
        back = 0
        if len(ch):
            h0, n0 = hand_depths(ctx, P, ch)
            h1, n1 = hand_depths(ctx, P2, ch)
            for a_, b_ in _runs(np.isin(np.arange(len(P)), ch), 1):
                sel = (ch >= a_) & (ch <= b_)
                if ((h1[sel] > np.maximum(h0[sel], 0.015) + 0.005) | (n1[sel] > np.maximum(n0[sel], 0.003) + 0.003)).any():
                    P2[a_:b_ + 1] = P[a_:b_ + 1]
                    back += b_ - a_ + 1
                else:
                    extra.append({"window": f"whip{len(extra)}", "adopted": True, "core": [int(a_), int(b_)], "ramp": 0, "ramp_l": 0,
                                  "ramp_r": 0, "hold": "fix2_whip", "t": [round(a_ / HZ, 2), round(b_ / HZ, 2)]})
        rep["whips"] = {"speed_repair": sp, "frames_changed": int(len(ch)), "frames_put_back": int(back)}
        P = P2
    P, rep["collar_reclamp"] = collar_reclamp(ctx["fit_lead"], ctx["P_in"], P)
    rep["frames_changed_vs_base"] = int((np.abs(P - P0).max((1, 2)) > 0).sum())
    rep["collar_deg_max_vs_input"] = round(float(collar_change_deg(ctx["P_in"], P).max()), 3)
    f = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P), ctx["p99"])
    rep["speed_flags_final"] = {k_: int(v_.sum()) for k_, v_ in f.items()}
    rep["seconds"] = round(time.time() - t0, 1)
    np.savez_compressed(d / "poses_merged.npz", lead=P)
    res = [{"window": w["window"], "adopted": True, "core": list(w["core"]), "ramp": 0, "ramp_l": int(w["core"][0] - w["span"][0]),
            "ramp_r": int(w["span"][1] - w["core"][1]), "hold": f"fix2_{w['kind']}", "t": w["t"]} for w in kept] + extra
    write_json(d / "refine.json", {"what": "the v43 fix round 2's kept windows (zouk_contacts.fix2_pass / fix2_merge)", "results": res})
    write_json(d / "report.json", py_json(rep))
    log(f"fix2: {len(kept)}/{len(rep['windows'])} windows kept, {rep['frames_changed_vs_base']} frames changed vs {base} ({rep['seconds']} s)")
    if write:
        rep["written"] = fix1_write(stage, base)
        write_json(d / "report.json", py_json(rep))
    return rep


# ------------------------------------------------------------------------------------------------ v43 fix round 3 (2026-10-23, verify H2 round 0)
#
# The independent check of fix 2 (verify H2 round 0) found 2:30 only partly fixed: on 150.2-156.37 s the fix-1 pose was kept (every fix-2
# way vetoed) - his palm at / above her neck joint for 2.9 s (150.67-153.6 s, up to 4.8 cm in her neck at 152.47-152.93 s; the input 0),
# then on her front (153.53-155.4 s); the same high hand at 240.3-241.5 s (4.4 cm in her neck). Their geometry: a close hold face to face,
# his right shoulder ~45-55 cm in front of her spine 3 - her shoulder blade (z -11 cm) is at his straight arm's reach (the 150.0-s pose has a
# straight elbow), her left back edge (x +17-20, z -6..0 cm) within it. Round 3 (on the fix-2 fit, frozen as v43h):
#   back edge  fix3_backedge: the hand of an anchor frame (150.0 / 240.0 s: palm flat on her upper-left back) carried in her spine-3 frame,
#              per frame laid on her skin (ray exit from her spine axis + lift) at the most-backward place round her left side (from her
#              left flank to her shoulder blade) the arm reaches with a legal elbow (not hyperextended past 5 deg, the wrist twist <= 88), at
#              shoulder-blade height (<= FIX3_Y_MAX above her spine 3; lower where only that reaches - never up toward her neck), the palm
#              turned to face her (smallest rotation), the anchor's fingers; the choice low-passed in time; raised-cosine ramps >= 0.4 s;
#              pushed out of her where it sinks in (fix2_push: down / back only out of her neck);
#   pops       fix3_crossfade: the f62 span's first frame (61.23 s) started 4 cm off fix 1 - a 0.5-s cross-fade from fix 1 into fix 2;
#              fix3_steps: the one-frame steps at 97.07 / 99.67 / 162.43 s low-passed (arm only, a depth guard);
#   roll       fix3_roll: his forearm's roll about its own axis (joint 19's global frame, the check's measure) low-passed inside the changed
#              spans to <= max(the input's p99, the input there) + FIX3_ROLL_MARGIN_DEG per frame; the wrist counter-rotated (the hand's
#              global rotation and every joint position unchanged).
# Output: stage "fix3" of curation_v43.json on v43h, the fit v43.

FIX3_STAGE = "fix3"
FIX3_BASE = "v43h"
FIX3_Y_MAX = 0.10                       # his palm at most this far above her spine 3 (her torso frame: up) - shoulder-blade height
FIX3_ROLL_MARGIN_DEG = 10.0
FIX3_BACKEDGE = ({"name": "b150", "core": (150.2, 155.6), "anchor": 150.0, "ramp": (0.4, 0.45)},
                 {"name": "b240", "core": (240.2, 241.45), "anchor": 240.0, "ramp": (0.4, 0.45)})
FIX3_XFADE = ({"name": "x61", "t0": 61.23, "frames": 16, "from": "v43g"},)      # f62's span start: fix 1 -> fix 2 over 0.53 s
FIX3_STEPS_S = (97.07, 99.67, 162.43)


def fix3_backedge(ctx: dict, spec: dict, cfg: "Fix1Cfg", thetas=tuple(range(-10, 65, 5)), ys=(0.10, 0.08, 0.06, 0.04, 0.02, 0.0),
                  y_pref: float = 0.08, fx_min: float = -5.0, miss_ok_m: float = 0.012, finger_up: float = 0.0,
                  lift_m: float = 0.025, push: str = "fix2", smooth: tuple = (9, 4.0)) -> dict:
    """the back-edge carry (see the section) on one window spec {"core": (t0, t1), "anchor": t, "ramp": (l, r) seconds} of the base
    (ctx["P_base"]) -> {"ks", "P" (W,55,3), "w" (W,) the blend weight, "stats", "VF", "kf", "theta", "y"}"""
    from scipy.ndimage import gaussian_filter1d, minimum_filter1d
    fl = ctx["fit_lead"]
    c0, c1 = (int(round(t * HZ)) for t in spec["core"])
    L0, L1 = (int(round(r * HZ)) for r in spec["ramp"])
    s0, s1 = c0 - L0, c1 + L1
    ks = np.arange(s0, s1 + 1)
    W = len(ks)
    kf = {k_: v_[ks] for k_, v_ in ctx["kf"].items()}
    Jf = kf["J"]
    VF = ctx["skinF"].verts(Jf, kf["R"])
    Pb = ctx["P_base"][ks].copy()
    hh = ctx["names"][ks] == "hand_hold"
    ia = int(round(spec["anchor"] * HZ)) - s0
    ob = posed(fl, ks, Pb)
    Wb, Gb = ob["J"][:, WRIST["right"]], ob["R"][:, WRIST["right"]]
    pcb, nb = hc.palm_centre(ob["J"], "right"), hc.palm_normal(ob["J"], "right")
    Ga = Gb[ia]
    off = Ga.T @ (pcb[ia] - Wb[ia])                     # the palm centre in the wrist's frame (the fingers do not move it)
    nl = Ga.T @ nb[ia]
    fl_ = Ga.T @ hc.finger_dir(ob["J"][ia:ia + 1], "right")[0]
    nl_ = _unit(nl - (nl @ fl_) * fl_)
    B_l = np.stack([fl_, nl_, np.cross(fl_, nl_)], 1)  # the hand's own basis (finger direction, palm normal) in the wrist's frame
    Rsp, O3 = kf["R"][:, SPINE3], Jf[:, SPINE3]
    Ea = Rsp[ia].T @ (ob["J"][ia, ELBOW["right"]] - O3[ia])
    E_hint = O3 + np.einsum("kij,j->ki", Rsp, Ea)        # ... and its elbow (the swivel's hint)
    Fs = spine_frame(Jf)
    ref = ctx["P_in"].copy()
    ref[ks] = Pb
    ref[ks, ELBOW["right"]] = Pb[ia, ELBOW["right"]]     # the anchor's elbow hinge
    flex_o, fixed_o = hc.finger_template(fl["betas"], "right", "open")
    fing = np.asarray(flex_o + fixed_o, float)          # the fingers open, flat on her (the anchor's curled thumb / index dug in 5-7 cm)
    # the thumb turned 45 deg toward the back of the hand (its tip 5 cm volar of the palm centre in the open template: it dug into her flank)
    from scipy.spatial.transform import Rotation as Ro
    fing[12] = Ro.from_matrix(rotvec_to_mat(np.radians(-45.0) * np.array([[1.0, 0.0, 0.0]]))[0] @ rotvec_to_mat(fing[12:13])[0]).as_rotvec()

    def place(th_deg, y, final=False):
        th = np.radians(np.broadcast_to(np.asarray(th_deg, float), (W,)))
        y = np.broadcast_to(np.asarray(y, float), (W,))
        d = np.cos(th)[:, None] * Fs[:, :, 0] - np.sin(th)[:, None] * Fs[:, :, 2]
        X = O3 + y[:, None] * Fs[:, :, 1] + 0.30 * d
        tp = fix2_radial_exit(ctx, Jf, VF, X, lift_m)
        if final and W > 5:
            # the final place low-passed in her torso frame (her mesh's raw exits / normals jittered the fingertips: added speed)
            Fr_ = torso_frame(Jf)
            tl = gaussian_filter1d(np.einsum("kji,kj->ki", Fr_, tp - O3), 3.0, axis=0, mode="nearest")
            tp = O3 + np.einsum("kij,kj->ki", Fr_, tl)
        _, tn = fix1_project(ctx["surfs"]["torso_lsh"], tp, VF, 0.0, Jf, sigma=4.0 if final else 0.0)
        # the hand flat on her: the palm facing her (-normal), the fingers along her skin round her toward her spine (a little up)
        tdir = -np.sin(th)[:, None] * Fs[:, :, 0] - np.cos(th)[:, None] * Fs[:, :, 2] + finger_up * Fs[:, :, 1]
        f = _unit(tdir - (tdir * tn).sum(1, keepdims=True) * tn)
        nn = _unit(-tn - (-tn * f).sum(1, keepdims=True) * f)
        B_t = np.stack([f, nn, np.cross(f, nn)], 2)
        G = B_t @ B_l.T
        Xw = tp - np.einsum("kij,j->ki", G, off)
        Pn, inf_ = arm_keep_hand(fl, ref, ref, ks, X_wrist=Xw, G21=G, E_hint=E_hint, twist_cap_deg=85.0)
        Pn = Pn[ks]
        Pn[:, 40:55] = fing
        oo = posed(fl, ks, Pn)
        perr = np.linalg.norm(hc.palm_centre(oo["J"], "right") - tp, axis=1)
        nerr = np.degrees(np.arccos(np.clip((hc.palm_normal(oo["J"], "right") * -tn).sum(1), -1.0, 1.0)))
        return Pn, np.maximum(inf_["wrist_miss_m"], perr), tp, nerr

    # ---- per frame the most-backward reachable place round her left side at shoulder-blade height
    cand = []
    for y in ys:
        for th in thetas:
            Pn, miss, tp, nerr = place(th, y)
            fx = elbow_angles(Pn, "right")["flexion"]
            tw = np.abs(wrist_angles(Pn, "right")["twist"])
            hgt = in_frame(tp, Jf)[:, 1]
            ok = (miss <= miss_ok_m) & (nerr <= 30.0) & (fx >= fx_min) & (tw <= 88.0) & (hgt <= FIX3_Y_MAX + 0.005)
            cand.append((th, y, ok, miss))
    score = np.full(W, -1e9)
    th_b, y_b = np.zeros(W), np.full(W, min(ys))
    miss_b = np.full(W, np.inf)
    for th, y, ok, miss in cand:
        s = th - 300.0 * abs(y - y_pref)
        better = ok & (s > score)
        score[better], th_b[better], y_b[better] = s, th, y
    none = score < -1e8
    if none.any():
        # nothing reaches legally: the place that misses least, low on her left flank
        for th, y, ok, miss in cand:
            if y <= 0.06 and th <= 20:
                b_ = none & (miss < miss_b)
                miss_b[b_], th_b[b_], y_b[b_] = miss[b_], th, y
    # low-passed (a lower envelope first: the smoothing must not carry a frame past its reach)
    th_s = gaussian_filter1d(minimum_filter1d(th_b, smooth[0], mode="nearest"), smooth[1], mode="nearest")
    y_s = gaussian_filter1d(minimum_filter1d(y_b, smooth[0], mode="nearest"), smooth[1], mode="nearest")
    Pn, miss, tp, nerr = place(th_s, y_s, final=True)
    Pn = smooth_rotations(Pn, (SHOULDER["right"], ELBOW["right"], WRIST["right"]), 2.0)
    # blend weights: 1 on the core, raised cosines over the ramps (never into a hand hold)
    w = np.zeros(W)
    for i, k in enumerate(ks):
        if c0 <= k <= c1:
            w[i] = 1.0
        elif k < c0:
            w[i] = 0.5 - 0.5 * np.cos(np.pi * (k - s0 + 1) / (L0 + 1))
        else:
            w[i] = 0.5 - 0.5 * np.cos(np.pi * (s1 + 1 - k) / (L1 + 1))
    w[hh] = 0.0
    P = Pb.copy()
    on = w > 0
    for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
        P[on, j] = _slerp_rotvec(Pb[on, j], Pn[on, j], w[on])
    stats = {"name": spec.get("name"), "core": [int(c0), int(c1)], "span": [int(s0), int(s1)], "anchor": int(ks[ia]),
             "no_legal_place": int((none & (w > 0)).sum()), "theta_deg": [round(float(th_s[w > 0].min()), 1), round(float(th_s[w > 0].max()), 1)],
             "y_cm": [round(float(y_s[w > 0].min()) * 100, 1), round(float(y_s[w > 0].max()) * 100, 1)],
             "miss_cm_p90": round(float(np.percentile(miss[w > 0], 90)) * 100, 2), "miss_cm_max": round(float(miss[w > 0].max()) * 100, 2),
             "palm_normal_err_deg_p90": round(float(np.percentile(nerr[w > 0], 90)), 1)}
    P_place = P.copy()
    # his hand out of her where it sinks in (the way out of her torso; out of her neck only down / back)
    sc = skin_contact(fl, P, ks, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    dp = on & ((sc["hand"] > 0.012) | (sc["neck"] > 0.004))
    if dp.any():
        Pp, stats["push"] = (fix3_push_level if push == "level" else fix2_push)(ctx, ks, P, dp, kf, VF)
        # the push faded with the blend weight in the ramps (an untapered push at the span's last frame popped 12 cm into the base)
        rm = (w > 0) & (w < 1)
        for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
            Pp[rm, j] = _slerp_rotvec(P[rm, j], Pp[rm, j], w[rm])
        Pp[w <= 0] = P[w <= 0]
        P = Pp
    return {"ks": ks, "P": P, "w": w, "stats": stats, "VF": VF, "kf": kf, "theta": th_s, "y": y_s, "P_place": P_place}


def fix3_push_level(ctx: dict, ks: np.ndarray, P: np.ndarray, sel: np.ndarray, kf: dict, VF: np.ndarray, rounds: int = 4,
                    tol_m: float = 0.012, margin_m: float = 0.006, max_m: float = 0.06, sigma: float = 2.0) -> tuple[np.ndarray, list]:
    """his hand out of her on the frames sel by moving the wrist LEVEL (her torso frame: no up / down) along her skin's outward normal at the
    palm (never up toward her neck - fix2_push's way out lifted the hand to neck height at 153 s), the hand's global rotation kept, the arm
    re-solved (arm_keep_hand); the push smoothed in time (sigma), <= max_m -> (P, per-round report)"""
    from scipy.ndimage import binary_dilation, gaussian_filter1d
    fl = ctx["fit_lead"]
    P = np.asarray(P, float).copy()
    W = len(ks)
    sel = np.asarray(sel, bool)
    rep = []
    o0 = posed(fl, ks, P)
    W0, G21 = o0["J"][:, WRIST["right"]].copy(), o0["R"][:, WRIST["right"]].copy()
    Fr = torso_frame(kf["J"])
    ref = ctx["P_in"].copy()
    ref[ks] = P
    dil = binary_dilation(sel, iterations=6)
    shift = np.zeros((W, 3))
    Pc = P.copy()
    for _ in range(rounds):
        o_ = posed(fl, ks, Pc)
        VH = ctx["skinL"].verts(o_["J"], o_["R"], ctx["ids"]["hand"][::2])
        dt = surface_depth(ctx["surfs"]["torso_arms"], VH, VF).max(1)
        dn = surface_depth(ctx["surfs"]["neck"], VH, VF).max(1)
        dep = np.maximum(dt, dn)
        bad = sel & (dep > tol_m)
        rep.append({"frames": int(bad.sum()), "depth_cm_max": round(float(dep[sel].max()) * 100, 2)})
        if not bad.any():
            break
        _, tn = fix1_project(ctx["surfs"]["torso_lsh"], hc.palm_centre(o_["J"], "right"), VF, 0.0, kf["J"], sigma=0.0)
        nl_ = np.einsum("kji,kj->ki", Fr, tn)
        nl_[:, 1] = 0.0
        way = _unit(np.einsum("kij,kj->ki", Fr, _unit(nl_)), tn)
        need = np.where(bad[:, None], (dep + margin_m)[:, None] * way, 0.0)
        tot = shift + need
        if W > 3:
            tot = gaussian_filter1d(tot, sigma, axis=0, mode="nearest")
            sc_ = np.linalg.norm(shift + need, axis=1) / np.maximum(np.linalg.norm(tot, axis=1), 1e-9)
            tot = tot * np.where(bad, np.maximum(sc_, 1.0), 1.0)[:, None]
        tot = tot * dil[:, None]
        tot *= np.minimum(1.0, max_m / np.maximum(np.linalg.norm(tot, axis=1), 1e-9))[:, None]
        shift = tot
        mv = np.linalg.norm(shift, axis=1) > 1e-5
        Pn, _ = arm_keep_hand(fl, ref, ref, ks[mv], X_wrist=W0[mv] + shift[mv], G21=G21[mv], twist_cap_deg=88.0)
        Pc = P.copy()
        Pc[mv] = Pn[ks[mv]]
    return Pc, rep


def forearm_roll_deg(fit_lead: dict, P: np.ndarray, ks: np.ndarray | None = None) -> np.ndarray:
    """(W,) his right forearm's roll per frame (deg, signed): joint 19's global rotation from frame to frame, its rotation vector's component
    about the forearm's own (rest elbow -> wrist) axis - the independent check's measure (verify H2_m6_twist); [0] = 0"""
    from scipy.spatial.transform import Rotation as Ro
    G = posed(fit_lead, ks, P)["R"][:, ELBOW["right"]]
    a = _unit(np.asarray(arm_rest(fit_lead["betas"])["r21"], float))
    D = np.einsum("kji,kjl->kil", G[:-1], G[1:])
    return np.r_[0.0, np.degrees(Ro.from_matrix(D).as_rotvec() @ a)]


def fix3_roll(ctx: dict, P: np.ndarray, allowed: np.ndarray, margin_deg: float = FIX3_ROLL_MARGIN_DEG, rounds: int = 12,
              sigma: float = 3.0, pad: int = 6) -> tuple[np.ndarray, dict]:
    """his forearm's roll about its own axis low-passed on the allowed frames (P full length (T,55,3)) where a frame's roll step exceeds
    max(the input's p99, the input's step there) + margin_deg: the cumulative roll smoothed there (gaussian sigma, +-pad frames), the elbow's
    local rotation turned about the forearm axis by the difference and the wrist's by its inverse (the hand's global rotation, the elbow
    and wrist positions unchanged) -> (P, {"frames", "max_before", "max_after", "spans"})"""
    from scipy.ndimage import binary_dilation, gaussian_filter1d
    from scipy.spatial.transform import Rotation as Ro
    fl = ctx["fit_lead"]
    P = np.asarray(P, float).copy()
    allowed = np.asarray(allowed, bool)
    rep = {"frames": 0, "spans": []}
    if not allowed.any():
        return P, rep
    a = _unit(np.asarray(arm_rest(fl["betas"])["r21"], float))
    ri_all = forearm_roll_deg(fl, ctx["P_in"])
    p99 = float(np.percentile(np.abs(ri_all), 99))
    rep["input_p99_deg"] = round(p99, 2)
    mx_b, mx_a = 0.0, 0.0
    for a0, b0 in _runs(allowed, 0):
        ks = np.arange(max(0, a0 - 1), min(len(P) - 1, b0 + 1) + 1)
        alw = allowed[ks].copy()
        r0 = forearm_roll_deg(fl, P[ks], ks)
        lim = np.maximum(p99, np.abs(ri_all[ks])) + margin_deg
        mx_b = max(mx_b, float(np.abs(r0).max()))
        if not (np.abs(r0) > lim).any():
            mx_a = max(mx_a, float(np.abs(r0).max()))
            continue
        psi = np.cumsum(np.radians(r0))
        ps = psi.copy()
        # a spin (the forearm turning a full turn more than it needs inside the span, e.g. +300 deg in 0.5 s at 61.23 s): unwound the short
        # way (a whole turn is the identity at the span's end), spread over the frames whose steps exceed the limit
        kk = int(np.round((psi[-1] - psi[0]) / (2 * np.pi)))
        if kk:
            ex = np.maximum(np.abs(r0) - lim, 0.0) * alw
            if ex.sum() > 0:
                ps = psi - 2 * np.pi * kk * np.cumsum(ex) / ex.sum()
        rnd = 0
        for rnd in range(rounds):
            st = np.r_[0.0, np.degrees(np.diff(ps))]
            bad = (np.abs(st) > lim) & alw
            if not bad.any():
                break
            m = binary_dilation(bad, iterations=pad) & alw
            m[0] = m[-1] = False                         # the span's ends stay (continuity with the frames outside)
            g = gaussian_filter1d(ps, sigma, mode="nearest")
            ps[m] = g[m]
        dl = (ps - psi + np.pi) % (2 * np.pi) - np.pi           # a whole turn is no turn (the span's unwound end stays bit-identical)
        ch = np.abs(dl) > 1e-9
        if ch.any():
            Tw = rotvec_to_mat(a[None] * dl[ch, None])
            R19 = rotvec_to_mat(P[ks[ch], ELBOW["right"]]) @ Tw
            R21 = np.transpose(Tw, (0, 2, 1)) @ rotvec_to_mat(P[ks[ch], WRIST["right"]])
            P[ks[ch], ELBOW["right"]] = Ro.from_matrix(R19).as_rotvec()
            P[ks[ch], WRIST["right"]] = Ro.from_matrix(R21).as_rotvec()
            rep["frames"] += int(ch.sum())
        r1 = forearm_roll_deg(fl, P[ks], ks)
        mx_a = max(mx_a, float(np.abs(r1).max()))
        rep["spans"].append({"t": [round(ks[0] / HZ, 2), round(ks[-1] / HZ, 2)], "max_before": round(float(np.abs(r0).max()), 1),
                             "max_after": round(float(np.abs(r1).max()), 1), "rounds": int(rnd), "unwound_turns": kk,
                             "turn_deg_max": round(float(np.degrees(np.abs(dl).max())), 1)})
    rep.update({"max_before": round(mx_b, 1), "max_after": round(mx_a, 1)})
    return P, rep



def fix3_crossfade(P: np.ndarray, P_from: np.ndarray, k0: int, n: int, fit_lead: dict | None = None) -> np.ndarray:
    """P (T,55,3) with frames k0 .. k0+n-1 cross-faded from P_from (weight 0 at k0 - 1) into P (weight 1 at k0 + n) on his right collar,
    arm and fingers (raised cosine). With fit_lead the fade runs in position space: the wrist, the elbow (the swivel's hint) and the hand's
    global rotation interpolated, the arm re-solved (arm_keep_hand; P's elbow hinge) - per-joint slerps of two arms whose rotations differ
    by a large swivel swing the hand far off between them (12 m/s fingertips at 61.6 s)"""
    P = np.asarray(P, float).copy()
    ks = np.arange(k0, k0 + n)
    w = 0.5 - 0.5 * np.cos(np.pi * (ks - k0 + 1) / (n + 1))
    Pn = P.copy()
    for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
        P[ks, j] = _slerp_rotvec(P_from[ks, j], Pn[ks, j], w)
    if fit_lead is None:
        return P
    from scipy.spatial.transform import Rotation as Ro
    oa, ob = posed(fit_lead, ks, P_from[ks]), posed(fit_lead, ks, Pn[ks])
    Xw = (1 - w)[:, None] * oa["J"][:, WRIST["right"]] + w[:, None] * ob["J"][:, WRIST["right"]]
    Eh = (1 - w)[:, None] * oa["J"][:, ELBOW["right"]] + w[:, None] * ob["J"][:, ELBOW["right"]]
    qa, qb = Ro.from_matrix(oa["R"][:, WRIST["right"]]).as_rotvec(), Ro.from_matrix(ob["R"][:, WRIST["right"]]).as_rotvec()
    G = rotvec_to_mat(_slerp_rotvec(qa, qb, w))
    ref = Pn.copy()
    ref[ks, COLLAR["right"]] = P[ks, COLLAR["right"]]
    P2, _ = arm_keep_hand(fit_lead, ref, ref, ks, X_wrist=Xw, G21=G, E_hint=Eh, twist_cap_deg=180.0)
    P[ks, SHOULDER["right"]] = P2[ks, SHOULDER["right"]]
    P[ks, ELBOW["right"]] = P2[ks, ELBOW["right"]]
    P[ks, WRIST["right"]] = P2[ks, WRIST["right"]]
    return P


def fix3_steps(ctx: dict, P: np.ndarray, times=FIX3_STEPS_S, half: int = 6, sigma: float = 1.5) -> tuple[np.ndarray, list]:
    """the one-frame steps at `times` (seconds): his arm (not the collar) low-passed on +-half frames - only frames already off the input;
    put back where that puts his hand deeper into her -> (P, report)"""
    P = np.asarray(P, float).copy()
    rep = []
    for t in times:
        k = int(round(t * HZ))
        a0, b0 = k - half, k + half
        seg = smooth_rotations(P[a0 - 8:b0 + 9], (SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)), sigma)[8:-8]
        ch = np.abs(P[a0:b0 + 1] - ctx["P_in"][a0:b0 + 1]).max((1, 2)) > 0
        fr = np.arange(a0, b0 + 1)[ch]
        if not len(fr):
            continue
        h0, n0 = hand_depths(ctx, P, fr)
        P2 = P.copy()
        P2[fr] = seg[ch]
        h1, n1 = hand_depths(ctx, P2, fr)
        okd = not ((h1 > np.maximum(h0, 0.012) + 0.004) | (n1 > np.maximum(n0, 0.003) + 0.002)).any()
        if okd:
            P = P2
        rep.append({"t": t, "frames": int(len(fr)), "kept": bool(okd)})
    return P, rep


def fix3_pass(base: str = FIX3_BASE, stage: str = FIX3_STAGE, device: str = "cuda", write: bool = True, only: tuple | None = None) -> dict:
    """fix round 3 (see the section) on the base (v43h = the fix-2 fit): the back-edge windows (FIX3_BACKEDGE, `only` their names), the
    f62 cross-fade (FIX3_XFADE), the one-frame steps (FIX3_STEPS_S), the speed rules on the back-edge spans, the forearm-roll low-pass on every
    frame fix 2 / fix 3 changed, the collar bounded again -> runs/<stage>/ (poses_merged.npz, refine.json, report.json); write: the fit v43
    = the base with these leader poses, curation_v43.json stage <stage>"""
    t0 = time.time()
    cfg = Fix1Cfg(device=device)
    ctx = fix1_context(base, device, collar=False)
    ctx["p99"] = speed_refs(ctx["fit_in"])
    P0 = ctx["P_base"].copy()
    P = P0.copy()
    T = len(P)
    rep = {"what": "v43 fix round 3 (zouk_contacts.fix3_pass, verify H2 round 0)", "base": base, "windows": []}
    res = []
    tapers = []
    spans = np.zeros(T, bool)
    for spec in FIX3_BACKEDGE:
        if only and spec["name"] not in only:
            continue
        r = fix3_backedge(ctx, spec, cfg)
        P[r["ks"]] = r["P"]
        ctx["P_base"] = P.copy()
        tapers.append((r["ks"], r["w"]))
        on = r["ks"][r["w"] > 0]
        spans[on] = True
        rep["windows"].append(r["stats"])
        c0, c1 = r["stats"]["core"]
        res.append({"window": spec["name"], "adopted": True, "core": [int(c0), int(c1)], "ramp": 0, "ramp_l": int(c0 - r["ks"][0]),
                    "ramp_r": int(r["ks"][-1] - c1), "hold": "fix3_backedge", "t": [round(c0 / HZ, 2), round(c1 / HZ, 2)]})
        log(f"fix3 {spec['name']}: {json.dumps(py_json(r['stats']))}")
    if spans.any():
        P, rep["speed_repair"] = speed_repair(ctx, P, rounds=8, half=6, sigma=3.0, allowed=spans)
    Pg = hc.full_pose(load_fit("v43g")["lead"])
    ch2 = (np.abs(P0 - Pg).max((1, 2)) > 0) & (np.abs(P - ctx["P_in"]).max((1, 2)) > 0)
    P, rep["roll_fix2"] = fix3_roll(ctx, P, ch2)                 # fix 2's forearm spins unwound before the cross-fade (its slerp flipped)
    for xf in FIX3_XFADE:
        k0 = int(round(xf["t0"] * HZ))
        Pf = hc.full_pose(load_fit(xf["from"])["lead"])
        P = fix3_crossfade(P, Pf, k0, int(xf["frames"]), ctx["fit_lead"])
        res.append({"window": xf["name"], "adopted": True, "core": [k0, k0 + int(xf["frames"]) - 1], "ramp": 0, "ramp_l": 0, "ramp_r": 0,
                    "hold": "fix3_xfade", "t": [round(k0 / HZ, 2), round((k0 + int(xf["frames"]) - 1) / HZ, 2)]})
    P, rep["steps"] = fix3_steps(ctx, P)
    for st in rep["steps"]:
        k = int(round(st["t"] * HZ))
        if st["kept"]:
            res.append({"window": f"step{st['t']}", "adopted": True, "core": [k - 6, k + 6], "ramp": 0, "ramp_l": 0, "ramp_r": 0,
                        "hold": "fix3_step", "t": [round((k - 6) / HZ, 2), round((k + 6) / HZ, 2)]})
    # the forearm roll on every frame fix 2 or fix 3 changed (v43g -> v43h, v43h -> now)
    alw = ((np.abs(P0 - Pg).max((1, 2)) > 0) | (np.abs(P - P0).max((1, 2)) > 0)) & (np.abs(P - ctx["P_in"]).max((1, 2)) > 0)
    P, rep["roll"] = fix3_roll(ctx, P, alw)
    for sp_ in rep["roll"]["spans"] + rep["roll_fix2"]["spans"]:
        if sp_["turn_deg_max"] > 0:
            a_, b_ = int(round(sp_["t"][0] * HZ)), int(round(sp_["t"][1] * HZ))
            res.append({"window": f"roll{sp_['t'][0]}", "adopted": True, "core": [a_, b_], "ramp": 0, "ramp_l": 0, "ramp_r": 0,
                        "hold": "fix3_roll", "t": sp_["t"]})
    # the back-edge ramps faded once more against the base (the push / speed steps inside a window left its last frame 6 cm off the base)
    for ks_, w_ in tapers:
        rm = (w_ > 0) & (w_ < 1)
        for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
            P[ks_[rm], j] = _slerp_rotvec(P0[ks_[rm], j], P[ks_[rm], j], w_[rm])
    P, rep["collar_reclamp"] = collar_reclamp(ctx["fit_lead"], ctx["P_in"], P)
    ch = np.abs(P - P0).max((1, 2)) > 0
    rep["frames_changed_vs_base"] = int(ch.sum())
    rep["changed_runs_s"] = [[round(a_ / HZ, 2), round(b_ / HZ, 2)] for a_, b_ in _runs(ch, 0)]
    rep["collar_deg_max_vs_input"] = round(float(collar_change_deg(ctx["P_in"], P).max()), 3)
    f = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P), ctx["p99"])
    rep["speed_flags_final"] = {k_: int(v_.sum()) for k_, v_ in f.items()}
    f0 = fix1_speed_flags(arm_points(ctx["fit_lead"], ctx["P_in"]), arm_points(ctx["fit_lead"], P0), ctx["p99"])
    rep["speed_flags_base"] = {k_: int(v_.sum()) for k_, v_ in f0.items()}
    rep["seconds"] = round(time.time() - t0, 1)
    d = OUT / "runs" / stage
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "poses_merged.npz", lead=P)
    write_json(d / "refine.json", {"what": "the v43 fix round 3's kept windows (zouk_contacts.fix3_pass)", "results": res})
    write_json(d / "report.json", py_json(rep))
    log(f"fix3: {len(res)} windows, {rep['frames_changed_vs_base']} frames changed vs {base} ({rep['seconds']} s)")
    if write:
        rep["written"] = fix1_write(stage, base)
        write_json(d / "report.json", py_json(rep))
    return rep


# ------------------------------------------------------------------------------------------------ v43 patch 2:30 (2026-10-23, verify H2 round 1 F1)
#
# The check of fix 3 (verify H2 round 1, F1) found two short awkward stretches inside b150: the lift (152.83-153.3 s: fix2_push lifted the
# hand out of her up to 27 cm above her spine 3, beside her left shoulder / head, the fingers up) and the tail (155.3-155.95 s: the palm
# dropped to her waist, off her, where the input and the phones have it on her upper-left back edge). The patch, on the fix-3 fit frozen as
# v43i, touches only those two stretches (every other frame = v43i):
#   lift   P230_LIFT: the hand carried in her spine frame from its pose at the run's first frame to its pose at the last (both <= 10 cm
#          above spine 3): the palm centre in her cylindrical coordinates (angle round her, height, radius), the hand's rotation slerped,
#          the elbow hint carried, the arm re-solved (arm_keep_hand); the height capped at P230_Y_CAP; where it sinks in, the wrist pushed
#          LEVEL along her skin's normal (fix3_push_level: never up, never to her front);
#   tail   P230_TAIL: the b150 core ends at 155.2 s; a per-joint slerp into the INPUT arm over 155.2-156.07 s (a near-constant rate), the input arm, then back
#          into the base (= fix 2 = ~ the input there) before the hand hold at 156.9 s; the input palm's depth in her back edge accepted;
#   roll   the lift's wrist twist straight between its ends (p230_twist); the collar bound again on the patched frames.
# Output: stage "p230" of curation_v43.json on v43i, the fit v43.

P230_STAGE = "p230"
P230_BASE = "v43i"
P230_LIFT = (152.5, 153.6)
P230_TAIL = {"core_end": 155.2, "in": (155.2, 156.07), "hold_to": 156.13, "out_to": 156.67}
P230_Y_CAP = 0.10


def p230_slerp_cont(A: np.ndarray, B: np.ndarray, w: np.ndarray) -> np.ndarray:
    """(n,3) rotation vectors: the slerp from A to B by w along consecutive frames with the relative rotation A^-1 B kept on one quaternion
    hemisphere through time (a per-frame shortest path flips where the two rotations are ~180 deg apart: a 50-deg forearm-roll jump)"""
    from scipy.spatial.transform import Rotation as Ro
    rel = (Ro.from_rotvec(A).inv() * Ro.from_rotvec(B)).as_quat()            # x y z w
    i0 = int(np.argmin(np.abs(np.asarray(w, float) - 0.5)))                     # the shortest way where the blend is half way ...
    if rel[i0, 3] < 0:
        rel[i0] = -rel[i0]
    for i in list(range(i0 + 1, len(rel))) + list(range(i0 - 1, -1, -1)):       # ... continued through time both ways
        jn = i - 1 if i > i0 else i + 1
        if rel[i] @ rel[jn] < 0:
            rel[i] = -rel[i]
    v = rel[:, :3]
    sv = np.linalg.norm(v, axis=1)
    ang = 2.0 * np.arctan2(sv, rel[:, 3])                                     # may exceed pi: the continuous way
    ax = v / np.maximum(sv, 1e-12)[:, None]
    return (Ro.from_rotvec(A) * Ro.from_rotvec(ax * (np.asarray(w, float) * ang)[:, None])).as_rotvec()


def p230_twist(P: np.ndarray, ks: np.ndarray, target_deg: np.ndarray, iters: int = 3) -> np.ndarray:
    """P with his right wrist's twist (wrist_angles) on ks set to target_deg: the elbow's local rotation turned about the forearm's rest axis
    by the difference, the wrist's by its inverse (R = T(a, twist) F: the twist changes exactly; the joint positions and the hand's global
    rotation stay)"""
    from scipy.spatial.transform import Rotation as Ro
    P = np.asarray(P, float).copy()
    a = _rest_axes("right")[0]
    for _ in range(iters):
        tw = wrist_angles(P[ks], "right")["twist"]
        d = np.radians(((tw - np.asarray(target_deg, float)) + 180.0) % 360.0 - 180.0)
        if np.abs(d).max() < 1e-6:
            break
        Tw = rotvec_to_mat(a[None] * d[:, None])
        P[ks, ELBOW["right"]] = Ro.from_matrix(rotvec_to_mat(P[ks, ELBOW["right"]]) @ Tw).as_rotvec()
        P[ks, WRIST["right"]] = Ro.from_matrix(np.transpose(Tw, (0, 2, 1)) @ rotvec_to_mat(P[ks, WRIST["right"]])).as_rotvec()
    return P


def p230_carry(ctx: dict, P: np.ndarray, ka: int, kb: int, y_cap: float = P230_Y_CAP, lift_m: float = 0.025) -> tuple[np.ndarray, dict]:
    """the lift (see the section): frames ka+1 .. kb-1 of P (T,55,3) carried in her spine frame from frame ka's hand to frame kb's -> (P, stats)"""
    from scipy.spatial.transform import Rotation as Ro
    fl = ctx["fit_lead"]
    P = np.asarray(P, float).copy()
    ks = np.arange(ka, kb + 1)
    Jf = ctx["kf"]["J"][ks]
    Fs, O3 = spine_frame(Jf), Jf[:, SPINE3]
    o = posed(fl, ks, P[ks])
    pc, Wr, G, E = hc.palm_centre(o["J"], "right"), o["J"][:, WRIST["right"]], o["R"][:, WRIST["right"]], o["J"][:, ELBOW["right"]]
    loc = lambda X: np.einsum("kji,kj->ki", Fs, X - O3)          # noqa: E731 (world -> her spine frame)
    pl, el = loc(pc), loc(E)
    s = 0.5 - 0.5 * np.cos(np.pi * (ks - ka) / (kb - ka))
    ang = np.arctan2(pl[:, 2], pl[:, 0])
    a0, a1 = ang[0], ang[0] + ((ang[-1] - ang[0] + np.pi) % (2 * np.pi) - np.pi)
    r = np.hypot(pl[:, 0], pl[:, 2])
    A_ = (1 - s) * a0 + s * a1
    R_ = (1 - s) * r[0] + s * r[-1] + lift_m * np.sin(np.pi * s)      # a little off her mid-run (her side bulges between the two ends)
    Y_ = np.minimum((1 - s) * pl[0, 1] + s * pl[-1, 1], y_cap)
    pc_n = O3 + np.einsum("kij,kj->ki", Fs, np.stack([R_ * np.cos(A_), Y_, R_ * np.sin(A_)], 1))
    gl = np.einsum("kji,kjl->kil", Fs, G)                          # the hand's rotation in her spine frame
    qa, qb = Ro.from_matrix(gl[0]).as_rotvec(), Ro.from_matrix(gl[-1]).as_rotvec()
    G_n = np.einsum("kij,kjl->kil", Fs, rotvec_to_mat(_slerp_rotvec(np.tile(qa, (len(ks), 1)), np.tile(qb, (len(ks), 1)), s)))
    off = (1 - s)[:, None] * (G[0].T @ (pc[0] - Wr[0])) + s[:, None] * (G[-1].T @ (pc[-1] - Wr[-1]))
    Xw = pc_n - np.einsum("kij,kj->ki", G_n, off)
    Eh = O3 + np.einsum("kij,kj->ki", Fs, (1 - s)[:, None] * el[0] + s[:, None] * el[-1])
    inner = ks[1:-1]
    # the reference arm slerped between the two ends (its elbow hinge -> the forearm's roll carried with the hand; the base's own roll
    # through the lift turned the forearm ~75 deg and put the wrist twist past its cap - a 9-m/s fingertip flip)
    ref = P.copy()
    for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"]):
        ref[inner, j] = _slerp_rotvec(np.tile(P[ka, j], (len(inner), 1)), np.tile(P[kb, j], (len(inner), 1)), s[1:-1])
    Pn, inf = arm_keep_hand(fl, ref, ref, inner, X_wrist=Xw[1:-1], G21=G_n[1:-1], E_hint=Eh[1:-1], twist_cap_deg=180.0)
    for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"]):
        P[inner, j] = Pn[inner, j]
    # the wrist's twist straight from one end's to the other's (the re-solved forearm's roll put it past 88 deg mid-run: a fingertip flip)
    t0_, t1_ = wrist_angles(P[[ka, kb]], "right")["twist"]
    P = p230_twist(P, inner, t0_ + s[1:-1] * (((t1_ - t0_) + 180.0) % 360.0 - 180.0))
    P[inner, 40:55] = _slerp_rotvec(np.tile(P[ka, 40:55], (len(inner), 1)), np.tile(P[kb, 40:55], (len(inner), 1)),
                                    np.repeat(s[1:-1], 15)).reshape(len(inner), 15, 3)
    return P, {"t": [round(ka / HZ, 2), round(kb / HZ, 2)], "y_cm": [round(float(Y_.min()) * 100, 1), round(float(Y_.max()) * 100, 1)],
               "wrist_miss_cm_max": round(float(np.max(inf["wrist_miss_m"])) * 100, 2)}


def p230_pass(base: str = P230_BASE, stage: str = P230_STAGE, device: str = "cuda", write: bool = True) -> dict:
    """the 2:30 patch (see the section) on the base (v43i = the fix-3 fit) -> runs/<stage>/ (poses_merged.npz, refine.json, report.json);
    write: the fit v43 = the base with these leader poses, curation_v43.json stage <stage>"""
    t0 = time.time()
    ctx = fix1_context(base, device, collar=False)
    fl = ctx["fit_lead"]
    P0 = ctx["P_base"].copy()
    P = P0.copy()
    T = len(P)
    rep = {"what": "v43 patch 2:30 (zouk_contacts.p230_pass, verify H2 round 1 F1)", "base": base}
    # ---- the lift
    ka, kb = (int(round(t * HZ)) for t in P230_LIFT)
    P, rep["lift"] = p230_carry(ctx, P, ka, kb)
    lk = np.arange(ka + 1, kb)
    kf = {k_: v_[lk] for k_, v_ in ctx["kf"].items()}
    VF = ctx["skinF"].verts(kf["J"], kf["R"])
    sc = skin_contact(fl, P[lk], lk, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    dp = (sc["hand"] > 0.012) | (sc["neck"] > 0.004)
    rep["lift"]["sink_frames"] = int(dp.sum())
    if dp.any():
        Pp, rep["lift"]["push"] = fix3_push_level(ctx, lk, P[lk], dp, kf, VF, max_m=0.04)
        # the push faded in / out at the run's ends (the carry's own weights: 0 at ka / kb)
        s = np.sin(np.pi * (lk - ka) / (kb - ka))
        for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"]):
            Pp[:, j] = _slerp_rotvec(P[lk, j], Pp[:, j], s)
        P[lk] = Pp
    # ---- the tail: the b150 core ends, the input arm, back to the base before the hand hold
    k_in0, k_in1 = (int(round(t * HZ)) for t in P230_TAIL["in"])
    k_h1, k_o1 = int(round(P230_TAIL["hold_to"] * HZ)), int(round(P230_TAIL["out_to"] * HZ))
    tk = np.arange(k_in0, k_o1 + 1)
    w = np.ones(len(tk))
    m_in, m_out = tk < k_in1, tk > k_h1
    # the way in: a near-constant rate (cosine ends of 4 frames) - the two forearms differ by ~200 deg of roll, a raised cosine peaked at 16 deg / frame
    u = (tk[m_in] - (k_in0 - 1)) / (k_in1 - (k_in0 - 1))
    e = 4.0 / (k_in1 - (k_in0 - 1))
    w[m_in] = np.where(u < e, e * (1 - np.cos(np.pi * u / (2 * e))) * 2 / np.pi, np.where(u > 1 - e, 1 - e * (1 - np.cos(np.pi * (1 - u) / (2 * e))) * 2 / np.pi,
                       e * 2 / np.pi + (u - e) * (1 - 4 * e / np.pi) / (1 - 2 * e)))
    w[m_out] = 0.5 + 0.5 * np.cos(np.pi * (tk[m_out] - k_h1) / (k_o1 + 1 - k_h1))
    assert not (ctx["names"][tk] == "hand_hold").any(), "the tail must end before the hand hold"
    # a per-joint slerp (his collar, arm, fingers) from the base into the input: both arms are smooth there, so the slerp's forearm roll
    # and wrist twist are too (a position-space blend wound the twist through 180 deg: the two arms' forearms differ by ~100 deg of roll)
    for j in (COLLAR["right"], SHOULDER["right"], ELBOW["right"], WRIST["right"], *range(40, 55)):
        P[tk, j] = p230_slerp_cont(P0[tk, j], ctx["P_in"][tk, j], w)
    # ... its wrist moved onto the straight path between the two arms' wrists (the slerp sagged the hand to her waist mid-way), the
    # slerp's hand rotation and elbow side kept (arm_keep_hand)
    oa, ob, os_ = posed(fl, tk, P0[tk]), posed(fl, tk, ctx["P_in"][tk]), posed(fl, tk, P[tk])
    Xw = (1 - w)[:, None] * oa["J"][:, WRIST["right"]] + w[:, None] * ob["J"][:, WRIST["right"]]
    tw_s = wrist_angles(P[tk], "right")["twist"]
    P2, _ = arm_keep_hand(fl, P, P, tk, X_wrist=Xw, E_hint=os_["J"][:, ELBOW["right"]], twist_cap_deg=180.0)
    for j in (SHOULDER["right"], ELBOW["right"], WRIST["right"]):
        P[tk, j] = P2[tk, j]
    P = p230_twist(P, tk, tw_s)                      # the slerp's wrist twist back (the re-solve rolled the forearm 30 deg / frame)
    # ... and the forearm's roll spread more evenly through the way in (the two forearms differ by ~240 deg of roll: a raised profile
    # peaked at 14 deg / frame); the wrist's twist takes the difference
    # half way between the two (all of it even: 7.6 deg / frame but the wrist twisted to 110 deg mid-way)
    kk = np.arange(k_in0 - 1, k_in1 + 1)
    cr0 = np.cumsum(forearm_roll_deg(fl, P[kk], kk))
    tgt = 0.5 * cr0 + 0.5 * cr0[-1] * np.arange(len(kk)) / (len(kk) - 1)
    for _ in range(3):
        cr = np.cumsum(forearm_roll_deg(fl, P[kk], kk))
        dl = tgt - cr
        tw_ = wrist_angles(P[kk[1:-1]], "right")["twist"]
        P = p230_twist(P, kk[1:-1], tw_ - dl[1:-1])
    kf = {k_: v_[tk] for k_, v_ in ctx["kf"].items()}
    VF = ctx["skinF"].verts(kf["J"], kf["R"])
    sc = skin_contact(fl, P[tk], tk, kf, VF, ctx["skinL"], ctx["ids"], ctx["surfs"])
    rep["tail"] = {"t": [round(tk[0] / HZ, 2), round(tk[-1] / HZ, 2)], "input_on": [round(k_in1 / HZ, 2), round(k_h1 / HZ, 2)],
                   "hand_cm_max": round(float(sc["hand"].max()) * 100, 2), "neck_cm_max": round(float(sc["neck"].max()) * 100, 2),
                   "palm_gap_cm_max": round(float(sc["palm_gap"].max()) * 100, 2)}
    # ---- the forearm roll, the collar, nothing outside the patch
    patch = np.zeros(T, bool)
    patch[lk] = True
    patch[tk] = True
    # the lift's wrist twist straight between its ends once more (the push re-solved the arm: its forearm roll wound it the long way); no
    # fix3_roll (it unwound a "whole turn" of the tail's ~180-deg slerp roll into a 24-deg step at its first frame)
    t0_, t1_ = wrist_angles(P0[[ka, kb]], "right")["twist"]
    sl = 0.5 - 0.5 * np.cos(np.pi * (lk - ka) / (kb - ka))
    P = p230_twist(P, lk, t0_ + sl * (((t1_ - t0_) + 180.0) % 360.0 - 180.0))
    P, rep["collar_reclamp"] = collar_reclamp(fl, ctx["P_in"], P)
    rl = forearm_roll_deg(fl, P)
    rep["roll"] = {"max_deg_lift": round(float(np.abs(rl[ka:kb + 2]).max()), 1), "max_deg_tail": round(float(np.abs(rl[tk[0]:tk[-1] + 2]).max()), 1)}
    P[~patch] = P0[~patch]
    ch = np.abs(P - P0).max((1, 2)) > 0
    rep["frames_changed_vs_base"] = int(ch.sum())
    rep["changed_runs_s"] = [[round(a_ / HZ, 2), round(b_ / HZ, 2)] for a_, b_ in _runs(ch, 0)]
    rep["collar_deg_max_vs_input"] = round(float(collar_change_deg(ctx["P_in"], P).max()), 3)
    rep["seconds"] = round(time.time() - t0, 1)
    res = [{"window": "lift", "adopted": True, "core": [int(lk[0]), int(lk[-1])], "ramp": 0, "ramp_l": 0, "ramp_r": 0, "hold": "p230_lift",
            "t": [round(lk[0] / HZ, 2), round(lk[-1] / HZ, 2)]},
           {"window": "tail", "adopted": True, "core": [int(tk[0]), int(tk[-1])], "ramp": 0, "ramp_l": 0, "ramp_r": 0, "hold": "p230_tail",
            "t": [round(tk[0] / HZ, 2), round(tk[-1] / HZ, 2)]}]
    d = OUT / "runs" / stage
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "poses_merged.npz", lead=P)
    write_json(d / "refine.json", {"what": "the v43 patch 2:30's windows (zouk_contacts.p230_pass)", "results": res})
    write_json(d / "report.json", py_json(rep))
    log(f"p230: {rep['frames_changed_vs_base']} frames changed vs {base} ({rep['seconds']} s)")
    if write:
        rep["written"] = fix1_write(stage, base)
        write_json(d / "report.json", py_json(rep))
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
    p.add_argument("--fit-out", default="v4", help="the fit written (v43: curation_v43.json)")
    p = sub.add_parser("gate")
    p.add_argument("--tag", default="pass3")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--no-coco", action="store_true")
    p.add_argument("--tags", default=None, help="comma list of the passes whose adopted windows the gate tests (default all2,pass2,pass3)")
    p.add_argument("--base-tag", default=None, help="the final pass: the reference motion runs/<base-tag>/ (e.g. inst); vetoes in stages")
    p.add_argument("--exact", action="store_true", help="also the exact mesh check of the candidate (GPU)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--vis", action="store_true", help="the closing pass: the 2D checks count only the phones that see the candidate's joint")
    p.add_argument("--fit-out", default="v4", help="the candidate fit (v43: curation_v43.json, her torso / shoulders / upper arms)")
    p = sub.add_parser("make-base")
    p.add_argument("--src", default="inst2")
    p.add_argument("--out", default="fin2")
    p.add_argument("--despike", default="[[165.73, 166.0, \"body\"], [347.43, 347.63, \"arm\"]]", help="json [[a_s, b_s, arm|body], ...]")
    p = sub.add_parser("validate-final")
    p.add_argument("--fits", default="input,v3,v4")
    p.add_argument("--changed-vs", default="v3")
    p.add_argument("--out", default=None, help="the table's stem under v4/contacts (default validate_final): <out>.json + <out>.md")
    p = sub.add_parser("before-after")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--extra", default="118,47,155")
    p = sub.add_parser("unity-motion")
    p.add_argument("--fit", default="v4", help="v4 (v4/smplx_contact) or v43 (v4/smplx_contact_v43 -> unity_motion_v43/)")
    p = sub.add_parser("plan-v43")
    p.add_argument("--out", default="plan_v43.json")
    p.add_argument("--cfg", default=None, help="json of ZoukCfg fields over V43_CFG")
    p = sub.add_parser("validate-v43")
    p.add_argument("--fits", default="input,v3,v4,v43")
    sub.add_parser("cache-check")
    p = sub.add_parser("prove-v43")
    p = sub.add_parser("last-pass", help="the last pass: spikes, deep frames, folds of the installed motion -> v4/smplx_contact")
    p.add_argument("--base", default="ec8e")
    p.add_argument("--out", default="v4")
    p.add_argument("--cfg", default=None, help="json of LastCfg fields")
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--only", default=None, help="a trial: only the windows around these seconds (comma list), report in last/trial.json")
    p.add_argument("--v43", action="store_true", help="the v43 last pass (her torso / shoulders / upper arms, torso holds only)")
    p.add_argument("--v43-parts", default=None, help="v43: the parts of her whose depth the pass works on (comma list of EXACT_PARTS)")
    p.add_argument("--v43-mask", default="torso_arms", help="v43: her surface mask (torso_arms | torso_arms_head)")
    p.add_argument("--stage", default="last43", help="v43: the curation stage / report folder name")
    p = sub.add_parser("fix1", help="v43 fix round 1: the collar bounded against the input + the analytic re-pose of the problem runs")
    p.add_argument("--base", default="v43")
    p.add_argument("--only", default=None, help="a trial: only the windows around these seconds (comma list) -> runs/<stage>/trial.json")
    p.add_argument("--write", action="store_true", help="also freeze the base as v43f and write the fit v43 (fix1-write)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--stage", default=FIX1_STAGE)
    p.add_argument("--cfg", default=None, help="json of Fix1Cfg fields")
    p.add_argument("--part", default=None, help="i/n: only the window groups g % n == i -> runs/<stage>/poses_part<i>.npz (then fix1-merge)")
    p = sub.add_parser("fix1-merge", help="join the parts of a parallel fix1 pass, speed repair, write")
    p.add_argument("--parts", type=int, required=True)
    p.add_argument("--base", default="v43")
    p.add_argument("--stage", default=FIX1_STAGE)
    p.add_argument("--write", action="store_true")
    p.add_argument("--device", default="cuda")
    p = sub.add_parser("fix1-write", help="runs/<stage>/poses_merged.npz -> the fit v43 (the base frozen as v43f)")
    p.add_argument("--stage", default=FIX1_STAGE)
    p.add_argument("--base", default="v43")
    p = sub.add_parser("fix2", help="v43 fix round 2: the owner's times (1:30 lean arm, 2:30 input mix, neck) on the frozen fix-1 fit (v43g)")
    p.add_argument("--base", default=FIX2_BASE)
    p.add_argument("--part", default=None, help="i/n: only the window groups g % n == i -> runs/<stage>/poses_part<i>.npz (then fix2-merge)")
    p.add_argument("--stage", default=FIX2_STAGE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-front", action="store_true", help="only the owner's ranges (not the remaining front runs)")
    p.add_argument("--only", default=None, help="only the windows around these seconds (comma list)")
    p.add_argument("--kinds", default=None, help="only these window kinds (comma list of 1m30, 2m30, neck240, f54, f62, f107, f322)")
    p.add_argument("--out-part", type=int, default=None, help="write runs/<stage>/poses_part<this>.npz (default: the part's i)")
    p.add_argument("--start", default=None, help="continue on runs/<stage>/poses_part<this>.npz (its kept windows in place)")
    p = sub.add_parser("fix3", help="v43 fix round 3 (verify H2 round 0): 2:30 / 240.5 s back-edge carry, the 61.23-s cross-fade, steps, "
                                    "the forearm roll; writes the fit v43 on v43h")
    p.add_argument("--base", default=FIX3_BASE)
    p.add_argument("--stage", default=FIX3_STAGE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", default=None, help="only these back-edge windows (comma list of b150, b240)")
    p.add_argument("--no-write", action="store_true")
    p = sub.add_parser("p230", help="v43 patch 2:30 (verify H2 round 1 F1): the b150 lift capped low on her back edge, the tail into the "
                                    "input arm; writes the fit v43 on v43i")
    p.add_argument("--base", default=P230_BASE)
    p.add_argument("--stage", default=P230_STAGE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-write", action="store_true")
    p = sub.add_parser("fix2-merge", help="join the parts of a fix2 pass, the whips, the collar bound; write the fit v43 on the base")
    p.add_argument("--parts", required=True, help="n (parts 0..n-1) or a comma list in merge order (a later part wins)")
    p.add_argument("--base", default=FIX2_BASE)
    p.add_argument("--stage", default=FIX2_STAGE)
    p.add_argument("--no-write", action="store_true")
    p.add_argument("--device", default="cuda")
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
        print(json.dumps(py_json(apply_curation(a.tag, base_tag=a.base_tag, fit_out=a.fit_out)), indent=1)[:3000])
    elif a.cmd == "gate":
        r = run_gate(a.tag, a.rounds, not a.no_coco, tags=a.tags.split(",") if a.tags else None, base_tag=a.base_tag, exact=a.exact,
                     device=a.device, vis=a.vis, fit_out=a.fit_out)
        print(json.dumps(py_json({"vetoed": r["windows_vetoed"], "rounds": r["rounds"]}), indent=1)[:6000])
    elif a.cmd == "make-base":
        print(json.dumps(py_json(make_base(a.src, a.out, json.loads(a.despike))), indent=1))
    elif a.cmd == "validate-final":
        v = validate_final(tuple(a.fits.split(",")), a.changed_vs, path=(OUT / f"{a.out}.md") if a.out else None,
                           json_path=(OUT / f"{a.out}.json") if a.out else None)
        print(validate_final_md(v))
    elif a.cmd == "validate":
        v = validate()
        print(json.dumps(py_json(v["table"]), indent=1)[:8000])
    elif a.cmd == "before-after":
        before_after_sheets(a.n, [int(x) for x in a.extra.split(",")] if a.extra else [])
    elif a.cmd == "unity-motion":
        print(json.dumps(py_json(export_motion(a.fit)), indent=1))
    elif a.cmd == "plan-v43":
        plan_v43(ZoukCfg(**{**V43_CFG, **(json.loads(a.cfg) if a.cfg else {})}), a.out)
    elif a.cmd == "validate-v43":
        v = validate_v43(tuple(a.fits.split(",")))
        print(validate_v43_md(v))
    elif a.cmd == "cache-check":
        print(json.dumps(py_json(cache_check()), indent=1))
    elif a.cmd == "prove-v43":
        print(json.dumps(py_json(prove_v43_export()), indent=1))
    elif a.cmd == "last-pass":
        r = last_pass(a.base, a.out, LastCfg(**json.loads(a.cfg)) if a.cfg else None, write=not a.no_write,
                      only=[float(x) for x in a.only.split(",")] if a.only else None, v43=a.v43,
                      v43_parts=tuple(a.v43_parts.split(",")) if a.v43_parts else None, v43_mask=a.v43_mask, stage=a.stage)
        print(json.dumps(py_json({k: r[k] for k in ("step1_frames_changed", "frames_changed_vs_base", "seconds")}), indent=1))
    elif a.cmd == "fix1":
        part = tuple(int(x) for x in a.part.split("/")) if a.part else None
        r = fix1_pass(a.base, Fix1Cfg(**json.loads(a.cfg)) if a.cfg else None, write=a.write and not part,
                      only=[float(x) for x in a.only.split(",")] if a.only else None, device=a.device, stage=a.stage, part=part)
        keys = ("collar", "frames_changed_vs_base", "collar_deg_max_vs_input", "speed_repair", "whips", "speed_flags_final", "collar_reclamp", "seconds")
        print(json.dumps(py_json({k: r.get(k) for k in keys}), indent=1))
    elif a.cmd == "fix1-merge":
        r = fix1_merge(a.parts, a.base, a.stage, a.write, a.device)
        keys = ("frames_changed_vs_base", "collar_deg_max_vs_input", "speed_repair", "whips", "speed_flags_final", "collar_reclamp", "seconds")
        print(json.dumps(py_json({k: r.get(k) for k in keys}), indent=1))
    elif a.cmd == "fix1-write":
        print(json.dumps(py_json(fix1_write(a.stage, a.base)), indent=1))
    elif a.cmd == "fix2":
        part = tuple(int(x) for x in a.part.split("/")) if a.part else None
        r = fix2_pass(a.base, part, a.stage, a.device, front=not a.no_front, only=[float(x) for x in a.only.split(",")] if a.only else None,
                      kinds=tuple(a.kinds.split(",")) if a.kinds else None, out_part=a.out_part, start=a.start)
        print(json.dumps(py_json([{k: w[k] for k in ("kind", "t", "adopted", "variant", "why")} for w in r["windows"]]), indent=1))
    elif a.cmd == "fix3":
        r = fix3_pass(a.base, a.stage, a.device, not a.no_write, tuple(a.only.split(",")) if a.only else None)
        keys = ("windows", "frames_changed_vs_base", "changed_runs_s", "collar_deg_max_vs_input", "steps", "roll", "speed_flags_base",
                "speed_flags_final", "collar_reclamp", "seconds")
        print(json.dumps(py_json({k: r.get(k) for k in keys}), indent=1))
    elif a.cmd == "p230":
        r = p230_pass(a.base, a.stage, a.device, not a.no_write)
        print(json.dumps(py_json({k: r.get(k) for k in ("lift", "tail", "roll", "collar_reclamp", "frames_changed_vs_base", "changed_runs_s",
                                                         "collar_deg_max_vs_input", "seconds")}), indent=1))
    elif a.cmd == "fix2-merge":
        r = fix2_merge(int(a.parts) if a.parts.isdigit() else a.parts.split(","), a.base, a.stage, not a.no_write, a.device)
        keys = ("frames_changed_vs_base", "collar_deg_max_vs_input", "whips", "speed_flags_final", "collar_reclamp", "seconds")
        print(json.dumps(py_json({k: r.get(k) for k in keys}), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
