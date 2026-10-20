r"""mono4d track "kimodo" (2026-10-19): can NVIDIA Kimodo fill a held-out second of REAL zouk motion?

Kimodo (github.com/nv-tlabs/kimodo, Apache-2.0 code; weights nvidia/Kimodo-SOMA-RP-v1.2, NVIDIA Open Model License, not
gated) is a kinematic motion-diffusion model on NVIDIA's uniform SOMA skeleton (somaskel30). It takes no video, so the
test is in-betweening: from the 8-phone MAMMA reference fit of larissa_kadu_38 (smplx/{lead,follow}.npz) the second
59.0-60.0 s is held out; Kimodo gets full-body keyframes on 58.0-59.0 s and 60.0-61.0 s (variants below), generates 5
samples, and the samples - mapped back to the dancers' own SMPL-X skeletons - are scored on the held-out second against
the reference, next to SLERP / linear / cubic interpolation of the same gap. Constraint-only: the text encoder
(LLM2Vec on the gated Meta-Llama-3-8B) is replaced by a zero stub; the prompt is '' (Kimodo zeroes empty prompts itself).

SMPL-X <-> SOMA (route "R2" of docs/TMR_SOMA_PLAN.md 3.2, implemented here): NVIDIA ships no SOMA -> SMPL-X pose
converter (soma-retargeter targets humanoid robots; SOMA-X has SMPL-family -> SOMA PoseInversion and identity
conversion, not the reverse), so both directions are a rotation copy:
    forward   G_soma[k] = G_smplx[map k] @ A_k (A_k = constant rest alignment: the SOMA rest bone onto the dancer's SMPL-X
              rest bone, + a second axis), FK on the uniform somaskel30 rest skeleton, Hips = s * pelvis + (0, dy, 0)
              (s = rest hip->ankle height ratio per dancer, dy = one floor offset per dancer and take so the take's
              5th-percentile ToeBase height is 0.03 m, Kimodo's own stance)
    inverse   G_smplx[j] = G_soma[k] @ A_k^T, pelvis = (Hips - (0, dy, 0)) / s, SMPL-X FK with the dancer's own rest
              offsets (from the reference fit: constant to 5e-8 m), neck->head bone along SOMA Neck1->Head
The round trip reference -> SOMA -> SMPL-X is exact by construction (checked: 'prep' prints it), so every error the
scores show is Kimodo's (or the baseline's), not the converter's.

Variants (each 5 samples, seed 0, 100 DDIM steps, cfg [text 2, constraint 2], post-processing off AND on for the same
samples):
    dense        full-body keyframes on all 60 context frames (as asked; out of Kimodo's training distribution of
                 1-20 sparse keyframes)
    dense_root   + the dense 2D smoothed-root path through the gap (ORACLE: the root path of the held-out second)
    sparse       18 keyframes (9 per side, denser next to the gap): in distribution
    sparse_root  + the oracle root path

    python -m dancecap.mono4d.cand_kimodo prep            # R2 conversion -> mono4d/kimodo/input/gapfill_inputs.npz
    python -m dancecap.mono4d.cand_kimodo job             # dancecap/remote/cand_kimodo/pod_job.json
    (then: python -m dancecap remote run --jobs ../dancecap/remote/cand_kimodo/pod_job.json --max-cost 3 --gpus ...)
    python -m dancecap.mono4d.cand_kimodo score [--session SID]   # inverse + baselines + scores -> mono4d/kimodo/
    python -m dancecap.mono4d.cand_kimodo render          # side-by-side stick figures + curves -> mono4d/kimodo/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

from ..take import PACKAGE_ROOT
from .candidates import score_arrays
from .holdout import procrustes

TAKE = "larissa_kadu_38"
WINDOW = (58.0, 61.0)
GAP = (59.0, 60.0)
WORK = PACKAGE_ROOT / "work" / TAKE
OUT = WORK / "mono4d" / "kimodo"
INPUT = OUT / "input" / "gapfill_inputs.npz"
POD_OUT = OUT / "pod_out"
JOB_DIR = PACKAGE_ROOT / "remote" / "cand_kimodo"
ROLES = ("lead", "follow")
NAME = "kimodo_gapfill"

KIMODO_REPO = "https://github.com/nv-tlabs/kimodo"
KIMODO_COMMIT = "e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954"     # main, 2026-10-15 ("remove unused tests")
KIMODO_HF = "nvidia/Kimodo-SOMA-RP-v1.2"
KIMODO_HF_REV = "0e3c89024d896064b6851201373f7834784a386b"     # HF model sha at the research read (2026-10-19)
SOMA30_BLOB = "9e319f6fa3cb4959ed3221040b0ae945ec9646c7"         # git blob of kimodo/assets/skeletons/somaskel30/joints.p
SOMA30_URL = f"https://raw.githubusercontent.com/nv-tlabs/kimodo/{KIMODO_COMMIT}/kimodo/assets/skeletons/somaskel30/joints.p"

# SMPL-X 55-joint kinematic tree (body 0-21, jaw 22, eyes 23-24, left hand 25-39, right hand 40-54)
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15,
                 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
                 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53]
L_MIDDLE1, R_MIDDLE1 = 28, 43
SOMA_NAMES = ["Hips", "Spine1", "Spine2", "Chest", "Neck1", "Neck2", "Head", "Jaw", "LeftEye", "RightEye",
              "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHandThumbEnd", "LeftHandMiddleEnd",
              "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHandThumbEnd", "RightHandMiddleEnd",
              "LeftLeg", "LeftShin", "LeftFoot", "LeftToeBase", "RightLeg", "RightShin", "RightFoot", "RightToeBase"]
SOMA_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28]
X, Y, Z = np.eye(3)
# SOMA joint k: (SMPL-X joint whose global rotation drives it, primary (soma bone, smplx bone), secondary)
#   a bone is (a, b) = rest[b] - rest[a]; an axis is a 3-vector (same in both rest poses: both face +Z, left = +X, Y up)
ALIGN = {
    0: (0, ((26, 22), (2, 1)), ((0, 1), (0, 3))),          # Hips: hip line exact, then the bone to Spine1 / spine1
    1: (3, ((1, 2), (3, 6)), X),                           # Spine1 -> Spine2 / spine1 -> spine2
    2: (6, ((2, 3), (6, 9)), X),                           # Spine2 -> Chest / spine2 -> spine3
    3: (9, ((3, 4), (9, 12)), X),                          # Chest -> Neck1 / spine3 -> neck (SOMA collars sit mid-line)
    4: (12, ((4, 5), (12, 15)), X),                        # Neck1 -> Neck2 / neck -> head
    5: (12, ((5, 6), (12, 15)), X),                        # Neck2 -> Head / neck -> head (one SMPL-X neck, two SOMA)
    6: (15, (("eyes", 6), ("eyes", 15)), ((9, 8), (24, 23))),  # Head: towards the eyes, eye line
    10: (13, ((10, 11), (13, 16)), Y), 11: (16, ((11, 12), (16, 18)), Y), 12: (18, ((12, 13), (18, 20)), Y),
    13: (20, ((13, 15), (20, L_MIDDLE1)), Y),
    16: (14, ((16, 17), (14, 17)), Y), 17: (17, ((17, 18), (17, 19)), Y), 18: (19, ((18, 19), (19, 21)), Y),
    19: (21, ((19, 21), (21, R_MIDDLE1)), Y),
    22: (1, ((22, 23), (1, 4)), Z), 23: (4, ((23, 24), (4, 7)), Z), 24: (7, ((24, 25), (7, 10)), X),
    26: (2, ((26, 27), (2, 5)), Z), 27: (5, ((27, 28), (5, 8)), Z), 28: (8, ((28, 29), (8, 11)), X),
}
LEAVES = [7, 8, 9, 14, 15, 20, 21, 25, 29]                 # no child in somaskel30: local identity
# inverse: SMPL-X body joint j (0-21, non-leaf in the 22-joint tree) <- SOMA joint k
SMPLX_FROM_SOMA = {0: 0, 3: 1, 6: 2, 9: 3, 12: 4, 13: 10, 14: 16, 16: 11, 17: 17, 18: 12, 19: 18,
                   1: 22, 2: 26, 4: 23, 5: 27, 7: 24, 8: 28}


def log(msg: str) -> None:
    print(f"[cand_kimodo {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------- skeletons


def soma_rest() -> np.ndarray:
    """somaskel30 rest joints (30, 3), Hips at 0 - Kimodo's asset at the pinned commit (git-blob checked, loaded with
    torch.load(weights_only=True)); cached as npz under mono4d/kimodo/assets/."""
    cache = OUT / "assets" / "somaskel30_joints.npz"
    if cache.exists():
        return np.load(cache)["joints"]
    raw = urllib.request.urlopen(SOMA30_URL, timeout=60).read()
    blob = hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest()
    if blob != SOMA30_BLOB:
        raise RuntimeError(f"somaskel30 joints.p blob {blob} != pinned {SOMA30_BLOB}")
    import io

    import torch
    j = torch.load(io.BytesIO(raw), weights_only=True).squeeze().numpy().astype(float)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, joints=j, names=np.array(SOMA_NAMES), parents=np.array(SOMA_PARENTS), source=SOMA30_URL, blob=blob)
    return j


def smplx_fk_rots(z) -> np.ndarray:
    """Global rotations (T, 55, 3, 3) from the stored SMPL-X parameters (axis-angle, flat_hand_mean hands)."""
    T = len(z["t"])
    aa = np.concatenate([z["global_orient"][:, None], z["body_pose"].reshape(T, 21, 3), z["jaw_pose"][:, None],
                         z["leye_pose"][:, None], z["reye_pose"][:, None], z["left_hand_pose"].reshape(T, 15, 3),
                         z["right_hand_pose"].reshape(T, 15, 3)], 1)
    L = Rot.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(T, 55, 3, 3)
    G = np.zeros_like(L)
    for j in range(55):
        G[:, j] = L[:, j] if SMPLX_PARENTS[j] < 0 else G[:, SMPLX_PARENTS[j]] @ L[:, j]
    return G


def smplx_rest_offsets(G: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, float]:
    """Rest offsets (55, 3) of each joint from its parent (0 for the root) = G_parent^T (J - J_parent), and their
    largest spread over frames (m): the check that the rotations reproduce the stored joints (V1)."""
    off = np.zeros((55, 3))
    spread = 0.0
    for c in range(1, 55):
        p = SMPLX_PARENTS[c]
        o = np.einsum("tji,tj->ti", G[:, p], J[:, c] - J[:, p])
        off[c] = o.mean(0)
        spread = max(spread, float(o.std(0).max()))
    return off, spread


def rest_positions(off: np.ndarray, parents: list[int]) -> np.ndarray:
    P = np.zeros((len(parents), 3))
    for c in range(1, len(parents)):
        P[c] = P[parents[c]] + off[c]
    return P


def _vec(spec, rest: np.ndarray, eye_pair: tuple[int, int]) -> np.ndarray:
    if isinstance(spec, np.ndarray):
        return spec
    a, b = spec
    pa = rest[list(eye_pair)].mean(0) if a == "eyes" else rest[a]
    pb = rest[b]
    return (pa - pb) if a == "eyes" else (pb - pa)


def _frame(p: np.ndarray, s: np.ndarray) -> np.ndarray:
    e1 = p / np.linalg.norm(p)
    e2 = s - np.dot(s, e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.stack([e1, e2, np.cross(e1, e2)], 1)


def alignments(soma: np.ndarray, smplx: np.ndarray) -> np.ndarray:
    """A (30, 3, 3): A_k maps the SOMA rest frame of joint k onto the SMPL-X rest frame of its driver (identity for
    leaves)."""
    A = np.tile(np.eye(3), (30, 1, 1))
    for k, (_, prim, sec) in ALIGN.items():
        ps, px = _vec(prim[0], soma, (8, 9)), _vec(prim[1], smplx, (23, 24))
        if isinstance(sec, np.ndarray):
            ss, sx = sec, sec
        else:
            ss, sx = _vec(sec[0], soma, (8, 9)), _vec(sec[1], smplx, (23, 24))
        A[k] = _frame(px, sx) @ _frame(ps, ss).T
    return A


def soma_fk(Gs: np.ndarray, hips: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Joint positions (T, 30, 3) of somaskel30 from global rotations and the Hips trajectory."""
    P = np.zeros(Gs.shape[:2] + (3,))
    P[:, 0] = hips
    for k in range(1, 30):
        p = SOMA_PARENTS[k]
        P[:, k] = P[:, p] + np.einsum("tij,j->ti", Gs[:, p], rest[k] - rest[p])
    return P


def to_soma(G: np.ndarray, pelvis: np.ndarray, A: np.ndarray, rest: np.ndarray, s: float, dy: float):
    """SMPL-X global rotations (T, 55) + pelvis (T, 3) -> SOMA global rotations, local rotations, positions, hips."""
    T = len(G)
    Gs = np.zeros((T, 30, 3, 3))
    for k in range(30):
        if k in LEAVES:
            Gs[:, k] = Gs[:, SOMA_PARENTS[k]]
        else:
            Gs[:, k] = G[:, ALIGN[k][0]] @ A[k]
    hips = s * pelvis + np.array([0.0, dy, 0.0])
    Ls = Gs.copy()
    for k in range(1, 30):
        Ls[:, k] = np.swapaxes(Gs[:, SOMA_PARENTS[k]], 1, 2) @ Gs[:, k]
    return Gs, Ls, soma_fk(Gs, hips, rest), hips


def _min_rot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(T, 3, 3) smallest rotations taking directions a onto b."""
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b)
    ang = np.arctan2(np.linalg.norm(v, axis=-1), np.sum(a * b, -1))
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    axis = np.where(n > 1e-12, v / np.maximum(n, 1e-12), 0.0)
    return Rot.from_rotvec(axis * ang[:, None]).as_matrix()


def from_soma(Gs: np.ndarray, Ps: np.ndarray, A: np.ndarray, off: np.ndarray, s: float, dy: float) -> np.ndarray:
    """SOMA global rotations (T, 30) + positions -> the dancer's SMPL-X body joints (T, 22, 3)."""
    T = len(Gs)
    G = np.zeros((T, 22, 3, 3))
    for j, k in SMPLX_FROM_SOMA.items():
        G[:, j] = Gs[:, k] @ A[k].T
    # one SMPL-X neck bone for SOMA's two: point neck -> head along SOMA Neck1 -> Head
    G[:, 12] = _min_rot(np.einsum("tij,j->ti", G[:, 12], off[15]), Ps[:, 6] - Ps[:, 4]) @ G[:, 12]
    J = np.zeros((T, 22, 3))
    J[:, 0] = (Ps[:, 0] - np.array([0.0, dy, 0.0])) / s
    for c in range(1, 22):
        p = SMPLX_PARENTS[c]
        J[:, c] = J[:, p] + np.einsum("tij,j->ti", G[:, p], off[c])
    return J


# ------------------------------------------------------------------------------------------- prep


def grid_indices(t: np.ndarray) -> dict:
    w = np.nonzero((t >= WINDOW[0]) & (t < WINDOW[1]))[0]
    tw = t[w]
    gap = np.nonzero((tw >= GAP[0]) & (tw <= GAP[1]))[0]
    ctx = np.setdiff1d(np.arange(len(w)), gap)
    a, b = ctx[ctx < gap[0]], ctx[ctx > gap[-1]]
    # sparse: 9 keyframes per side, denser next to the gap (< 20 per constraint type, Kimodo's training range)
    sa = sorted({int(a[-1] - d) for d in (0, 2, 4, 8, 12, 16, 20, 24, 28) if a[-1] - d >= a[0]})
    sb = sorted({int(b[0] + d) for d in (0, 2, 4, 8, 12, 16, 20, 24, 28) if b[0] + d <= b[-1]})
    return {"win": w, "gap": gap, "ctx": ctx, "sparse": np.array(sa + sb)}


def prep() -> dict:
    rest = soma_rest()
    arrays, info = {"soma_rest": rest}, {"window": WINDOW, "gap": GAP, "roles": {}}
    for role in ROLES:
        z = np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True)
        t, J = np.asarray(z["t"], float), np.asarray(z["joints"], float)
        G = smplx_fk_rots(z)
        off, spread = smplx_rest_offsets(G, J)
        xr = rest_positions(off, SMPLX_PARENTS)
        A = alignments(rest, xr)
        s = float((rest[0, 1] - rest[[24, 28], 1].mean()) / (xr[0, 1] - xr[[7, 8], 1].mean()))
        _, _, P0, _ = to_soma(G, J[:, 0], A, rest, s, 0.0)
        dy = float(0.03 - np.percentile(P0[:, [25, 29], 1].min(1), 5))
        Gs, Ls, Ps, hips = to_soma(G, J[:, 0], A, rest, s, dy)
        back = from_soma(Gs, Ps, A, off, s, dy)
        rt = float(np.abs(back - J[:, :22]).max())
        bl = {SOMA_NAMES[k]: float(np.abs(np.linalg.norm(Ps[:, k] - Ps[:, SOMA_PARENTS[k]], axis=-1)
                                          - np.linalg.norm(rest[k] - rest[SOMA_PARENTS[k]])).max()) for k in range(1, 30)}
        ix = grid_indices(t)
        w = ix["win"]
        for key, val in (("pos", Ps[w]), ("grot", Gs[w]), ("lrot", Ls[w]), ("hips", hips[w]), ("t", t[w]),
                         ("A", A), ("smplx_off", off), ("smplx_joints", J[w, :22])):
            arrays[f"{role}_{key}"] = val
        for key in ("gap", "ctx", "sparse"):
            arrays[key] = ix[key]
        toe = Ps[:, [25, 29], 1].min(1)
        info["roles"][role] = {"s": s, "dy": dy, "smplx_fk_spread_m": spread, "roundtrip_max_m": rt,
                               "bone_len_err_max_m": max(bl.values()), "toe_p5_m": float(np.percentile(toe, 5)),
                               "hips_y_median_m": float(np.median(Ps[:, 0, 1])), "frames_window": int(len(w)),
                               "t_window": [float(t[w[0]]), float(t[w[-1]])],
                               "t_gap": [float(t[w[ix['gap'][0]]]), float(t[w[ix['gap'][-1]]])],
                               "quality_min_window": float(np.min(z["quality"][w])),
                               "n_gap": int(len(ix["gap"])), "n_ctx": int(len(ix["ctx"])), "sparse": ix["sparse"].tolist()}
        log(f"{role}: s {s:.4f} dy {dy:+.4f} m, SMPL-X FK spread {spread:.1e} m, round trip max {rt:.1e} m, "
            f"bone length err {max(bl.values()):.1e} m, window {len(w)} frames, gap {len(ix['gap'])}")
    INPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(INPUT, **arrays)
    info.update(input=str(INPUT), bytes=INPUT.stat().st_size, created=time.strftime("%Y-%m-%d %H:%M:%S"),
                route="R2 rotation copy (TMR_SOMA_PLAN 3.2), exact inverse", soma_rest_blob=SOMA30_BLOB)
    (INPUT.parent / "prep.json").write_text(json.dumps(info, indent=1))
    return info


# ------------------------------------------------------------------------------------------- pod job


def job(max_minutes: float = 30) -> Path:
    ws = lambda p: str(Path(p).resolve().relative_to(PACKAGE_ROOT.parent)).replace("\\", "/")  # noqa: E731
    d = {"schema": "dancecap.pod_job/1", "name": "cand_kimodo", "take": TAKE, "max_minutes": max_minutes, "needs": [],
         "inputs": [ws(INPUT), ws(INPUT.parent / "prep.json")],
         # this job needs no MAMMA code or cached wheels: a 11 MB bundle instead of 351 MB over the shared 2.5 MB/s uplink
         # (session 20261019-140212: the 351 MB upload timed out at 702 s while other sessions uploaded)
         "exclude": ["dancecap/work/runpod/pywheels", "dancecap/work/runpod/wheelhouse", "mamma-turing"],
         "env": {"KIMODO_COMMIT": KIMODO_COMMIT, "KIMODO_HF": KIMODO_HF, "KIMODO_HF_REV": KIMODO_HF_REV,
                 "CHECKPOINT_DIR": "/workspace/kimodo_cand/ckpt", "LOCAL_CACHE": "true",
                 "NVIDIA_TF32_OVERRIDE": "0", "TOKENIZERS_PARALLELISM": "false"},
         "commands": [
             {"name": "install", "timeout_min": 18, "cmd": "bash remote/cand_kimodo/install.sh"},
             {"name": "gapfill", "timeout_min": 8,
              "cmd": "/workspace/kimodo_cand/venv/bin/python remote/cand_kimodo/kimodo_gapfill_pod.py "
                     f"--inputs {ws(INPUT)[len('dancecap/'):]} --out {ws(POD_OUT)[len('dancecap/'):]} --samples 5 --seed 0"}],
         "outputs": [ws(POD_OUT) + "/"],
         "verify": [ws(POD_OUT / "gapfill.json")],
         "after_download": ["python -m dancecap.mono4d.cand_kimodo score", "python -m dancecap.mono4d.cand_kimodo render"],
         "notes": ["Kimodo constraint-only gap fill (no text encoder, no Llama); weights Kimodo-SOMA-RP-v1.2 (NVIDIA Open "
                   "Model License) downloaded on the pod at a pinned revision; uploads only joint arrays of the "
                   "reference fit (no frames, no video)"]}
    JOB_DIR.mkdir(parents=True, exist_ok=True)
    p = JOB_DIR / "pod_job.json"
    p.write_text(json.dumps(d, indent=1), newline="\n")
    return p


# ------------------------------------------------------------------------------------------- baselines + score


def _local_rots(z, w):
    T = len(z["t"])
    aa = np.concatenate([z["global_orient"][:, None], z["body_pose"].reshape(T, 21, 3)], 1)[w]
    return aa


def baselines(role: str, inp, gap: np.ndarray) -> dict:
    """SLERP of the SMPL-X local rotations + linear pelvis, linear joints, cubic Hermite joints across the gap."""
    z = np.load(WORK / "smplx" / f"{role}.npz", allow_pickle=True)
    t = np.asarray(z["t"], float)
    w = grid_indices(t)["win"]
    J = inp[f"{role}_smplx_joints"]
    off = inp[f"{role}_smplx_off"]
    a, b = int(gap[0] - 1), int(gap[-1] + 1)
    alpha = (gap - a) / (b - a)
    aa = _local_rots(z, w)
    out = {}
    # SLERP (local rotations) + linear pelvis -> SMPL-X FK with the dancer's offsets
    Lr = np.zeros((len(J), 22, 3, 3))
    for j in range(22):
        sl = Slerp([0.0, 1.0], Rot.from_rotvec(aa[[a, b], j]))
        Lr[:, j] = Rot.from_rotvec(aa[:, j]).as_matrix()
        Lr[gap, j] = sl(alpha).as_matrix()
    Gr = np.zeros_like(Lr)
    for j in range(22):
        Gr[:, j] = Lr[:, j] if SMPLX_PARENTS[j] < 0 else Gr[:, SMPLX_PARENTS[j]] @ Lr[:, j]
    Js = np.zeros_like(J)
    Js[:, 0] = J[:, 0]
    Js[gap, 0] = (1 - alpha)[:, None] * J[a, 0] + alpha[:, None] * J[b, 0]
    for c in range(1, 22):
        Js[:, c] = Js[:, SMPLX_PARENTS[c]] + np.einsum("tij,j->ti", Gr[:, SMPLX_PARENTS[c]], off[c])
    Js[np.setdiff1d(np.arange(len(J)), gap)] = J[np.setdiff1d(np.arange(len(J)), gap)]
    out["slerp"] = Js
    Jl = J.copy()
    Jl[gap] = (1 - alpha)[:, None, None] * J[a] + alpha[:, None, None] * J[b]
    out["linear"] = Jl
    # cubic Hermite on positions with the boundary velocities (central differences inside each context)
    va, vb = (J[a] - J[a - 2]) / 2.0, (J[b + 2] - J[b]) / 2.0
    n = b - a
    u = alpha[:, None, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    Jc = J.copy()
    Jc[gap] = h00 * J[a] + h10 * n * va + h01 * J[b] + h11 * n * vb
    out["cubic"] = Jc
    return out


def motion_stats(J: np.ndarray, idx: np.ndarray, fps: float = 30.0) -> dict:
    """Mean joint speed (m/s) and acceleration (m/s^2) over idx (22 joints), and the head / wrists' path length (m)."""
    v = np.diff(J, axis=0) * fps
    acc = np.diff(J, 2, axis=0) * fps * fps
    iv = idx[idx < len(v)]
    ia = idx[(idx >= 1) & (idx - 1 < len(acc))] - 1
    sp = np.linalg.norm(v[iv], axis=-1)
    ac = np.linalg.norm(acc[ia], axis=-1)
    path = {nm: float(np.linalg.norm(np.diff(J[idx[0]:idx[-1] + 2, j], axis=0), axis=-1).sum())
            for nm, j in (("head", 15), ("l_wrist", 20), ("r_wrist", 21), ("l_ankle", 7), ("r_ankle", 8))}
    return {"speed_mean": float(sp.mean()), "acc_mean": float(ac.mean()), "path_m": path}


def heading_deg(J: np.ndarray) -> np.ndarray:
    """(T,) unwrapped hip-line heading (deg; 0 = facing +Z), Kimodo's definition on the SMPL-X hips."""
    h = J[:, 1] - J[:, 2]
    return np.degrees(np.unwrap(np.arctan2(h[:, 2], -h[:, 0])))


def _errs(J: np.ndarray, R: np.ndarray, idx: np.ndarray) -> dict:
    m = 1e3 * np.linalg.norm(J[idx] - R[idx], axis=-1).mean(-1)
    pa = np.array([1e3 * np.linalg.norm(procrustes(J[i], R[i]) - R[i], axis=-1).mean() for i in idx])
    hj, hr = heading_deg(J), heading_deg(R)
    hj += np.round((hr[idx[0] - 1] - hj[idx[0] - 1]) / 360.0) * 360.0   # same winding at the frame before idx
    mid = int(idx[len(idx) // 2])
    return {"mpjpe_mm": float(np.median(m)), "mpjpe_mean_mm": float(m.mean()), "mpjpe_max_mm": float(m.max()),
            "pa_mm": float(np.median(pa)), "pa_mean_mm": float(pa.mean()),
            "heading_err_mid_deg": float(hj[mid] - hr[mid]), "heading_err_max_deg": float(np.abs(hj[idx] - hr[idx]).max()),
            "turn_dir_ok": bool(abs(hj[mid] - hr[mid]) < 90.0)}


def _session_meta(sid: str | None) -> dict:
    root = PACKAGE_ROOT / "work" / "runpod" / "sessions"
    if sid is None:
        cands = sorted((p for p in root.glob("*") if (p / "result.json").exists()
                        and "cand_kimodo" in (p / "result.json").read_text(errors="ignore")), key=lambda p: p.stat().st_mtime)
        if not cands:
            return {}
        sid = cands[-1].name
    p = root / sid / "result.json"
    if not p.exists():
        return {"session": sid}
    r = json.loads(p.read_text())
    pod = r.get("pod") or {}
    return {"session": sid, "gpu": pod.get("gpu"), "cost_per_hr": pod.get("cost_per_hr"), "cost_usd": r.get("cost"),
            "result": {k: r.get(k) for k in ("cost", "minutes", "pod", "ledger") if k in r}}


def score(sid: str | None = None) -> dict:
    inp = np.load(INPUT, allow_pickle=True)
    gap, ctx = inp["gap"], inp["ctx"]
    res = json.loads((POD_OUT / "gapfill.json").read_text())
    prep_info = json.loads((INPUT.parent / "prep.json").read_text())
    out = {"name": NAME, "take": TAKE, "window": WINDOW, "gap": GAP, "pod": res.get("info"), "session": _session_meta(sid),
           "roles": {}, "notes": []}
    traj = {}
    for role in ROLES:
        R = inp[f"{role}_smplx_joints"]
        t = inp[f"{role}_t"]
        A, off = inp[f"{role}_A"], inp[f"{role}_smplx_off"]
        s, dy = prep_info["roles"][role]["s"], prep_info["roles"][role]["dy"]
        rows = {}
        bl = baselines(role, inp, gap)
        for nm, Jb in bl.items():
            traj[(role, nm, 0)] = Jb
            rows[nm] = {"gap": _errs(Jb, R, gap), "harness": score_arrays(t, Jb, "world", t, R, GAP),
                        "motion_gap": motion_stats(Jb, gap)}
        rows["reference"] = {"motion_gap": motion_stats(R, gap)}
        traj[(role, "reference", 0)] = R
        for var in res.get("variants", []):
            for pp in ("raw", "pp"):
                f = POD_OUT / f"{role}_{var}_{pp}.npz"
                if not f.exists():
                    continue
                k = np.load(f)
                Rc, o = k["canon_R"], k["canon_o"]           # world = Rc^T-inverse: X_c = Rc (X - o)
                Pw = np.einsum("ij,stnj->stni", Rc.T, k["posed_joints"]) + o
                Gw = np.einsum("ij,stnjk->stnik", Rc.T, k["global_rot_mats"])
                kf = ctx if "dense" in var else inp["sparse"]
                # +anchor only for the raw samples: Kimodo's own post-processing already hits the keyframes exactly
                gen = {f"{var}_{pp}": [], **({f"{var}_{pp}+anchor": []} if pp == "raw" else {})}
                for si in range(len(Pw)):
                    Js = from_soma(Gw[si], Pw[si], A, off, s, dy)
                    # +anchor (OURS, not Kimodo's post-processing): the context frames are the given keyframes, and the
                    # residual at the two frames bordering the gap is blended linearly out across the gap
                    a0, b0 = int(gap[0] - 1), int(gap[-1] + 1)
                    al = ((gap - a0) / (b0 - a0))[:, None, None]
                    Ja = R.copy()
                    Ja[gap] = Js[gap] - (1 - al) * (Js[a0] - R[a0]) - al * (Js[b0] - R[b0])
                    for key, J_ in ((f"{var}_{pp}", Js), (f"{var}_{pp}+anchor", Ja)):
                        if key not in gen:
                            continue
                        traj[(role, key, si)] = J_
                        e = _errs(J_, R, gap)
                        e["ctx"] = _errs(J_, R, ctx)
                        e["harness"] = score_arrays(t, J_, "world", t, R, GAP)
                        e["motion_gap"] = motion_stats(J_, gap)
                        # Kimodo-space keyframe adherence (SOMA joints, canonical frame)
                        e["soma_keyframe_mm"] = float(1e3 * np.linalg.norm(k["posed_joints"][si][kf] - k["target_pos"][kf],
                                                                           axis=-1).mean())
                        gen[key].append(e)
                for key, per in gen.items():
                    rows[key] = _sample_rows(per, traj, role, key, gap)
        out["roles"][role] = rows
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "trajectories.npz", **{f"{r}__{m}__{i}": v for (r, m, i), v in traj.items()},
                        t=inp["lead_t"], gap=gap, ctx=ctx, sparse=inp["sparse"])
    (OUT / "scores.json").write_text(json.dumps(out, indent=1, default=float))
    (OUT / "scores.md").write_text(table(out))
    return out


def _sample_rows(per: list, traj: dict, role: str, key: str, gap: np.ndarray) -> dict:
    mp = [p["mpjpe_mm"] for p in per]
    pa = [p["pa_mm"] for p in per]
    return {"samples": per, "n": len(per),
            "gap_mpjpe_mm_median_over_samples": float(np.median(mp)),
            "gap_mpjpe_mm_best_sample_ORACLE": float(np.min(mp)),
            "gap_mpjpe_mm_spread": [float(np.min(mp)), float(np.max(mp))],
            "gap_pa_mm_median_over_samples": float(np.median(pa)),
            "gap_pa_mm_best_sample_ORACLE": float(np.min(pa)),
            "ctx_mpjpe_mm_median_over_samples": float(np.median([p["ctx"]["mpjpe_mm"] for p in per])),
            "soma_keyframe_mm_median": float(np.median([p["soma_keyframe_mm"] for p in per])),
            "turn_dir_ok": f"{sum(p['turn_dir_ok'] for p in per)}/{len(per)}",
            "heading_err_mid_deg": [round(p["heading_err_mid_deg"], 1) for p in per],
            "sample_diversity_gap_mm": float(np.mean([1e3 * np.linalg.norm(
                traj[(role, key, i)][gap] - traj[(role, key, j)][gap], axis=-1).mean()
                for i in range(len(per)) for j in range(i + 1, len(per))])) if len(per) > 1 else None}


def table(sc: dict) -> str:
    lines = [f"# Kimodo gap fill, {TAKE}: held-out {GAP[0]:.1f}-{GAP[1]:.1f} s (30 frames), keyframes on "
             f"{WINDOW[0]:.1f}-{GAP[0]:.1f} and {GAP[1]:.1f}-{WINDOW[1]:.1f} s; reference = 8-phone fit", "",
             "Kimodo rows: median over 5 samples (best sample = oracle pick, in brackets). Errors on the gap frames, "
             "SMPL-X body joints 0-21, mm.", "",
             "| role | method | MPJPE (world) | PA-MPJPE | root-rel | pelvis | limb deg | ctx MPJPE | speed m/s (ref) | turn dir ok |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for role, rows in sc["roles"].items():
        ref_sp = rows["reference"]["motion_gap"]["speed_mean"]
        for m, r in rows.items():
            if m == "reference":
                continue
            if "samples" in r:
                h = [p["harness"] for p in r["samples"]]
                med = lambda key: float(np.median([x[key]["median"] for x in h]))  # noqa: E731
                lines.append(f"| {role} | kimodo {m} | {r['gap_mpjpe_mm_median_over_samples']:.0f} "
                             f"[{r['gap_mpjpe_mm_best_sample_ORACLE']:.0f}] | {r['gap_pa_mm_median_over_samples']:.0f} "
                             f"[{r['gap_pa_mm_best_sample_ORACLE']:.0f}] | {med('rr_mm'):.0f} | {med('pelvis_mm'):.0f} | "
                             f"{med('limb_deg'):.1f} | {r['ctx_mpjpe_mm_median_over_samples']:.0f} | "
                             f"{np.median([p['motion_gap']['speed_mean'] for p in r['samples']]):.2f} ({ref_sp:.2f}) | {r['turn_dir_ok']} |")
            else:
                h = r["harness"]
                lines.append(f"| {role} | {m} | {r['gap']['mpjpe_mm']:.0f} | {r['gap']['pa_mm']:.0f} | "
                             f"{h['rr_mm']['median']:.0f} | {h['pelvis_mm']['median']:.0f} | {h['limb_deg']['median']:.1f} | "
                             f"0 | {r['motion_gap']['speed_mean']:.2f} ({ref_sp:.2f}) | {'yes' if r['gap']['turn_dir_ok'] else 'NO'} |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------- render


BONES22 = [(0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13),
           (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21)]


def render(method: str | None = None, fps: int = 15) -> list[Path]:
    """Side-by-side stick figures (reference | SLERP | Kimodo) for the 3-s window, both dancers in the world frame
    (front view = camera 06 direction is not used: a fixed world view, x right, y up), + error / height curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter

    sc = json.loads((OUT / "scores.json").read_text())
    tr = np.load(OUT / "trajectories.npz")
    gap = tr["gap"]
    t = tr["t"]
    if method is None:  # the Kimodo variant with the best median-over-samples gap MPJPE (lead + follow)
        ks = [m for m in sc["roles"]["lead"] if "samples" in sc["roles"]["lead"][m]]
        method = min(ks, key=lambda m: sum(sc["roles"][r][m]["gap_mpjpe_mm_median_over_samples"] for r in ROLES))
    # the sample closest to the median error (not the oracle best) for each role
    pick = {}
    for r in ROLES:
        e = [p["mpjpe_mm"] for p in sc["roles"][r][method]["samples"]]
        pick[r] = int(np.argsort(e)[len(e) // 2])
    cols = [("reference (8 phones)", "reference", {r: 0 for r in ROLES}), ("SLERP baseline", "slerp", {r: 0 for r in ROLES}),
            (f"Kimodo {method} (median sample)", method, pick)]
    out = []
    outs = OUT / "render"
    outs.mkdir(parents=True, exist_ok=True)
    # one view per dancer: looking along the dancer's mean facing direction (front view), centred on its own pelvis
    view = {}
    for r in ROLES:
        R = tr[f"{r}__reference__0"]
        hip = (R[:, 1] - R[:, 2])[:, [0, 2]].mean(0)          # left-right axis (x, z)
        hip /= np.linalg.norm(hip)
        view[r] = (np.r_[R[:, 0, [0, 2]].mean(0)], hip)

    def proj(J, r):
        c, u = view[r]
        h = (J[..., 0] - c[0]) * u[0] + (J[..., 2] - c[1]) * u[1]   # dancer's left on the image right (front view)
        return h, J[..., 1]

    colr = {"lead": "#1f4e79", "follow": "#c0392b"}
    fig, axs = plt.subplots(2, 3, figsize=(10.8, 10.2), dpi=100)
    lines = {}
    for ri, r in enumerate(ROLES):
        for ci, (title, m, pk) in enumerate(cols):
            ax = axs[ri, ci]
            ax.set_title(f"{r}: {title}", fontsize=10)
            ax.set_xlim(-1.0, 1.0)
            ax.set_ylim(-0.05, 1.95)
            ax.set_aspect("equal")
            ax.axhline(0, color="#999", lw=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if m != "reference":
                lines[(ci, r, "ref")] = [ax.plot([], [], color="#c8c8c8", lw=1.6)[0] for _ in BONES22]
            lines[(ci, r)] = [ax.plot([], [], color=colr[r], lw=2.4, solid_capstyle="round")[0] for _ in BONES22]
            lines[(ci, r, "head")] = ax.plot([], [], "o", color=colr[r], ms=8)[0]
    txt = fig.suptitle("")

    def draw(f):
        for ci, (title, m, pk) in enumerate(cols):
            for r in ROLES:
                h, v = proj(tr[f"{r}__{m}__{pk[r]}"][f], r)
                for li, (a, b) in zip(lines[(ci, r)], BONES22):
                    li.set_data([h[a], h[b]], [v[a], v[b]])
                lines[(ci, r, "head")].set_data([h[15]], [v[15]])
                if (ci, r, "ref") in lines:
                    hr, vr = proj(tr[f"{r}__reference__0"][f], r)
                    for li, (a, b) in zip(lines[(ci, r, "ref")], BONES22):
                        li.set_data([hr[a], hr[b]], [vr[a], vr[b]])
        ing = gap[0] <= f <= gap[-1]
        txt.set_text(f"larissa_kadu_38  t = {t[f]:.2f} s   {'HELD-OUT GAP (generated)' if ing else 'keyframes (given)'}"
                     f"\nfront view of each dancer; grey = reference")
        txt.set_color("#b03a2e" if ing else "#222")
        return []

    try:
        writer = FFMpegWriter(fps=fps, bitrate=2500)
        mp4 = outs / f"side_by_side_{method}.mp4"
        with writer.saving(fig, str(mp4), dpi=100):
            for f in range(len(t)):
                draw(f)
                writer.grab_frame()
        out.append(mp4)
    except Exception as e:  # no ffmpeg: animated GIF
        log(f"ffmpeg writer failed ({e}); writing a GIF")
        gif = outs / f"side_by_side_{method}.gif"
        writer = PillowWriter(fps=fps)
        with writer.saving(fig, str(gif), dpi=80):
            for f in range(len(t)):
                draw(f)
                writer.grab_frame()
        out.append(gif)
    # filmstrip of the gap: rows = methods, columns = 7 instants across the held-out second
    inst = np.linspace(gap[0] - 1, gap[-1] + 1, 7).round().astype(int)
    fig2, axs2 = plt.subplots(2 * len(cols), len(inst), figsize=(15, 2.9 * 2 * len(cols)), dpi=80)
    for di, r in enumerate(ROLES):
        for mi, (title, m, pk) in enumerate(cols):
            ri = di * len(cols) + mi
            for ci, f in enumerate(inst):
                ax = axs2[ri, ci]
                h, v = proj(tr[f"{r}__{m}__{pk[r]}"][f], r)
                if m != "reference":
                    hr, vr = proj(tr[f"{r}__reference__0"][f], r)
                    for a, b in BONES22:
                        ax.plot([hr[a], hr[b]], [vr[a], vr[b]], color="#c8c8c8", lw=1.4)
                for a, b in BONES22:
                    ax.plot([h[a], h[b]], [v[a], v[b]], color=colr[r], lw=1.9)
                ax.plot([h[15]], [v[15]], "o", color=colr[r], ms=5)
                ax.set_xlim(-0.9, 0.9)
                ax.set_ylim(-0.05, 1.95)
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if ri == 0:
                    ax.set_title(f"{t[f]:.2f} s" + (" (key)" if not (gap[0] <= f <= gap[-1]) else ""), fontsize=10)
                if ci == 0:
                    ax.set_ylabel(f"{r}\n{title}", fontsize=8)
    fig2.suptitle(f"Held-out second {GAP[0]:.0f}-{GAP[1]:.0f} s, front view of each dancer: grey = reference", fontsize=12)
    fig2.tight_layout()
    p2 = outs / f"filmstrip_{method}.png"
    fig2.savefig(p2)
    out.append(p2)
    # curves: per-frame MPJPE and head / wrist heights
    fig3, axs3 = plt.subplots(2, 3, figsize=(15, 7.5), dpi=90)
    keys = ["slerp", "cubic", method]
    for ri, r in enumerate(ROLES):
        R = tr[f"{r}__reference__0"]
        ax = axs3[ri, 0]
        for m in keys:
            n = len(sc["roles"][r][m]["samples"]) if "samples" in sc["roles"][r][m] else 1
            for si in range(n):
                J = tr[f"{r}__{m}__{si}"]
                e = 1e3 * np.linalg.norm(J - R, axis=-1).mean(-1)
                ax.plot(t, e, lw=1.6 if n == 1 else 0.9, alpha=1 if n == 1 else 0.7,
                        label=(m if n == 1 else f"kimodo {m} (5 samples)") if si == 0 else None,
                        color={"slerp": "#888888", "cubic": "#e67e22"}.get(m, "#2e86c1"),
                        ls="--" if m == "slerp" else "-")
        ax.axvspan(t[gap[0]], t[gap[-1]], color="#f6d5d0", alpha=0.5)
        ax.set_title(f"{r}: MPJPE vs reference (mm)")
        ax.legend(fontsize=7)
        for ai, (j, nm) in enumerate(((15, "head height (m)"), (21, "right wrist height (m)"))):
            ax = axs3[ri, ai + 1]
            ax.plot(t, R[:, j, 1], color="k", lw=2.2, label="reference")
            ax.plot(t, tr[f"{r}__slerp__0"][:, j, 1], color="#888", lw=1.4, ls="--", label="slerp")
            n = len(sc["roles"][r][method]["samples"])
            for si in range(n):
                ax.plot(t, tr[f"{r}__{method}__{si}"][:, j, 1], color="#2e86c1", lw=0.9, alpha=0.7,
                        label=f"kimodo {method}" if si == 0 else None)
            ax.axvspan(t[gap[0]], t[gap[-1]], color="#f6d5d0", alpha=0.5)
            ax.set_title(f"{r}: {nm}")
            ax.legend(fontsize=7)
    fig3.tight_layout()
    p3 = outs / f"curves_{method}.png"
    fig3.savefig(p3)
    out.append(p3)
    plt.close("all")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.cand_kimodo")
    ap.add_argument("action", choices=["prep", "job", "score", "render", "table"])
    ap.add_argument("--session")
    ap.add_argument("--method")
    ap.add_argument("--max-minutes", type=float, default=30)
    a = ap.parse_args(argv)
    if a.action == "prep":
        print(json.dumps(prep(), indent=1))
    elif a.action == "job":
        print(job(a.max_minutes))
    elif a.action == "score":
        print(table(score(a.session)))
    elif a.action == "table":
        print(table(json.loads((OUT / "scores.json").read_text())))
    else:
        for p in render(a.method):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
