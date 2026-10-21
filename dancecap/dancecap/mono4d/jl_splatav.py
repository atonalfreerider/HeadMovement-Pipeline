"""john_lula LHM++ canonical PLYs -> skinned splat avatars (docs/SPLAT_AVATAR_FORMAT.md, `SPAV` v1) for the Unity proof.

The john_lula realism run saved only LHM++'s canonical Gaussians (`work/john_lula/realism/lhmpp/pod_out/gs/<role>_cano.ply`,
pose = LHM++'s `animate_gs_model` canonical pose, shaped by the betas it used) and no skinning. This tool makes an
APPROXIMATE avatar locally, at $0, so the Unity splat layer can be proven before the recap export (which carries LHM++'s
own weights and exact zero-pose means, `mono4d/recap_splatav.py`) arrives:

1. finds which canonical pose the PLY is in (zero pose / shoulders lowered / shoulders + hips) by the distance of the
   opaque, small Gaussians to the SMPL-X surface posed that way (the smallest median wins, reported);
2. skinning weights = inverse-distance blend of the 8 nearest posed SMPL-X vertices' LBS weights, top 4 renormalised
   (LHM++ samples a diffused voxel field instead: weights differ off the body, hair and loose clothes);
3. un-poses every Gaussian to the shaped ZERO pose (the format's canonical pose) with the inverse of its blended
   canonical skinning matrix; rotation q0 = quat(M_c)^-1 (x) q_c;
4. checks the result against LHM++'s own posed scene (`scene_key<k>.ply`, both dancers at pose key k): plain LBS of
   this file with the pod's pose for key k vs LHM++'s posed centres (median / p95 mm, written to the JSON `checks`);
5. writes `<out>/<role>.splatav` + `<role>.json` in Unity coordinates (export_unity's z mirror).

    python -m dancecap.mono4d.jl_splatav [--out head-movement/Assets/StreamingAssets/08_JohnLula/splats] [--key 3020]

Likeness data: the output goes to git-ignored StreamingAssets / work only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np

from ..take import WORKSPACE

ROLES = ("lead", "follow")
N_J = 55
C0 = 0.28209479177387814
MAGIC = b"SPAV"
RECORD = 120
HEADER = 64
REALISM = WORKSPACE / "dancecap" / "work" / "john_lula" / "realism" / "lhmpp"
MODEL = WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
DEFAULT_OUT = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets" / "08_JohnLula" / "splats"
MIRROR = np.array([1.0, 1.0, -1.0])

RECORD_TABLE = [["mean", "f32", 3], ["rotation_xyzw", "f32", 4], ["log_scale", "f32", 3], ["opacity_logit", "f32", 1],
                ["sh0", "f32", 3], ["joints", "u16", 4], ["weights", "f32", 4], ["hair", "u8", 1],
                ["reserved", "u8", 3], ["strand", "i32", 1], ["segment", "i32", 1], ["local_offset", "f32", 3],
                ["local_rotation_xyzw", "f32", 4]]
RECORD_DTYPE = np.dtype([("mean", "<f4", 3), ("rotation", "<f4", 4), ("log_scale", "<f4", 3), ("opacity", "<f4"),
                         ("sh0", "<f4", 3), ("joints", "<u2", 4), ("weights", "<f4", 4), ("hair", "u1"),
                         ("reserved", "u1", 3), ("strand", "<i4"), ("segment", "<i4"), ("local_offset", "<f4", 3),
                         ("local_rotation", "<f4", 4)])
assert RECORD_DTYPE.itemsize == RECORD


# ---------------------------------------------------------------------------------------------------- small maths

def read_ply(path: Path) -> dict:
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:end].decode().splitlines()
    assert "binary_little_endian" in lines[1], lines[1]
    n = int(next(ln for ln in lines if ln.startswith("element vertex")).split()[-1])
    props = [ln.split()[-1] for ln in lines if ln.startswith("property")]
    assert all(ln.split()[1] == "float" for ln in lines if ln.startswith("property")), "float PLY expected"
    a = np.frombuffer(raw[end:end + n * 4 * len(props)], "<f4").reshape(n, len(props))
    return {k: a[:, i].astype(np.float64) for i, k in enumerate(props)}


def rodrigues(rv: np.ndarray) -> np.ndarray:
    rv = np.asarray(rv, float).reshape(-1, 3)
    th = np.linalg.norm(rv, axis=1)[:, None, None]
    k = rv / np.maximum(th[:, :, 0], 1e-12)
    K = np.zeros((len(rv), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    return np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, xyzw."""
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def quat_conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def mat_to_quat(m: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> unit xyzw (Shepperd; works on the slightly non-orthonormal blended matrices too)."""
    m = np.asarray(m, float)
    sh = m.shape[:-2]
    m = m.reshape(-1, 3, 3)
    q = np.zeros((len(m), 4))
    tr = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cases = np.stack([tr, m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]], 1).argmax(1)
    for c in range(4):
        i = cases == c
        if not i.any():
            continue
        r = m[i]
        if c == 0:
            s = np.sqrt(np.maximum(tr[i] + 1.0, 1e-12)) * 2
            q[i] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s, 0.25 * s], 1)
        elif c == 1:
            s = np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-12)) * 2
            q[i] = np.stack([0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s,
                             (r[:, 2, 1] - r[:, 1, 2]) / s], 1)
        elif c == 2:
            s = np.sqrt(np.maximum(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-12)) * 2
            q[i] = np.stack([(r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s, (r[:, 1, 2] + r[:, 2, 1]) / s,
                             (r[:, 0, 2] - r[:, 2, 0]) / s], 1)
        else:
            s = np.sqrt(np.maximum(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-12)) * 2
            q[i] = np.stack([(r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s,
                             (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*sh, 4)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = np.moveaxis(np.asarray(q, float), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


class Smplx:
    """Plain numpy SMPL-X (no pose blendshapes, flat hands = zero hand pose): shaped rest + FK + skin matrices."""

    def __init__(self, path: Path = MODEL):
        m = np.load(path, allow_pickle=True)
        self.v_template = m["v_template"].astype(np.float64)
        self.shapedirs = m["shapedirs"].astype(np.float64)
        self.J_regressor = m["J_regressor"].astype(np.float64)
        self.weights = m["weights"].astype(np.float64)
        self.parents = m["kintree_table"][0].astype(np.int64)
        self.parents[0] = -1
        self.faces = m["f"].astype(np.int64)
        self.path = Path(path)

    def shaped(self, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = np.asarray(betas, float)
        v = self.v_template + np.einsum("vdk,k->vd", self.shapedirs[:, :, :len(b)], b)
        return v, self.J_regressor @ v

    def fk(self, J: np.ndarray, full_pose_aa: np.ndarray, transl=(0.0, 0.0, 0.0)) -> np.ndarray:
        """(55,3) axis-angle -> (55,4,4) joint world transforms G_k = [R_k | P_k]."""
        R = rodrigues(full_pose_aa)
        G = np.zeros((N_J, 4, 4))
        for k in range(N_J):
            p = self.parents[k]
            L = np.eye(4)
            L[:3, :3] = R[k]
            L[:3, 3] = J[k] + np.asarray(transl) if p < 0 else J[k] - J[p]
            G[k] = L if p < 0 else G[p] @ L
        return G

    @staticmethod
    def skin_mats(G: np.ndarray, J: np.ndarray) -> np.ndarray:
        """S_k = G_k T(-J_k)."""
        S = G.copy()
        S[:, :3, 3] = G[:, :3, 3] - np.einsum("kab,kb->ka", G[:, :3, :3], J)
        return S


def lbs(S: np.ndarray, joints: np.ndarray, weights: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Blend the 4 skin matrices per point: returns posed points (N,3) and blended matrices (N,4,4)."""
    M = np.einsum("nk,nkab->nab", weights, S[joints])
    return np.einsum("nab,nb->na", M[:, :3, :3], x) + M[:, :3, 3], M


def cano_pose_candidates() -> dict[str, np.ndarray]:
    def pose(sh: float, hip: float) -> np.ndarray:
        p = np.zeros((N_J, 3))
        p[16, 2], p[17, 2] = -sh, +sh  # left / right shoulder about z
        p[1, 2], p[2, 2] = +hip, -hip  # left / right hip about z
        return p
    d30, d15 = np.pi / 6, np.pi / 12
    return {"zero": pose(0, 0), "shoulders_30": pose(d30, 0), "shoulders_30_hips_15": pose(d30, d15),
            "shoulders_45": pose(np.pi / 4, 0)}


def knn(src: np.ndarray, dst: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from scipy.spatial import cKDTree
    d, i = cKDTree(dst).query(src, k=k)
    return d, i


# ---------------------------------------------------------------------------------------------------- the writer

def write_splatav(out_dir: Path, role: str, rec: np.ndarray, header: dict) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(rec)
    pre = MAGIC + struct.pack("<7I", 1, n, RECORD, 0, 1, HEADER, 0) + b"\0" * (HEADER - 32)
    assert len(pre) == HEADER
    data = pre + rec.astype(RECORD_DTYPE).tobytes()
    path = out_dir / f"{role}.splatav"
    path.write_bytes(data)
    header = dict(header)
    header.update({"data": path.name, "data_bytes": len(data), "data_sha256": hashlib.sha256(data).hexdigest()})
    (out_dir / f"{role}.json").write_text(json.dumps(header, indent=1))
    return header


def read_splatav(path: Path) -> tuple[dict, np.ndarray]:
    raw = Path(path).read_bytes()
    assert raw[:4] == MAGIC, raw[:4]
    version, n, rb, sh_rest, flags, hb = struct.unpack("<6I", raw[4:28])
    assert version == 1 and rb == RECORD and hb == HEADER, (version, rb, hb)
    rec = np.frombuffer(raw[HEADER:HEADER + n * RECORD], RECORD_DTYPE)
    return {"version": version, "count": n, "sh_rest": sh_rest, "flags": flags}, rec


# ---------------------------------------------------------------------------------------------------- one dancer

def convert(role: str, out_dir: Path, key: int, model: Smplx, report: dict, scene: dict | None,
            scene_offset: int) -> dict:
    t0 = time.time()
    pod = json.loads((REALISM / "pod_out" / "lhmpp_render_pod.json").read_text())
    info = pod["dancers"][role]
    betas = np.asarray(info["betas_used"], float)
    g = read_ply(REALISM / "pod_out" / "gs" / info["cano_ply"])
    xc = np.stack([g["x"], g["y"], g["z"]], 1)
    qc = np.stack([g["rot_1"], g["rot_2"], g["rot_3"], g["rot_0"]], 1)  # PLY w first -> xyzw
    qc /= np.linalg.norm(qc, axis=1, keepdims=True)
    log_scale = np.stack([g["scale_0"], g["scale_1"], g["scale_2"]], 1)
    opac = g["opacity"]
    sh0 = np.stack([g["f_dc_0"], g["f_dc_1"], g["f_dc_2"]], 1)
    n = len(xc)

    v_rest, J = model.shaped(betas)
    # 1. which canonical pose: surface distance of the opaque, small Gaussians
    core = (1 / (1 + np.exp(-opac)) > 0.5) & (np.exp(log_scale).max(1) < 0.012)
    fits = {}
    for name, p in cano_pose_candidates().items():
        S = model.skin_mats(model.fk(J, p), J)
        vp = np.einsum("vk,kab->vab", model.weights, S)
        vp = np.einsum("vab,vb->va", vp[:, :3, :3], v_rest) + vp[:, :3, 3]
        d, _ = knn(xc[core], vp, 1)
        fits[name] = round(float(np.median(d) * 1e3), 2)
    pose_name = min(fits, key=fits.get)
    pose_c = cano_pose_candidates()[pose_name]
    Gc = model.fk(J, pose_c)
    Sc = model.skin_mats(Gc, J)
    Wv = model.weights
    Mv = np.einsum("vk,kab->vab", Wv, Sc)
    v_c = np.einsum("vab,vb->va", Mv[:, :3, :3], v_rest) + Mv[:, :3, 3]

    # 2. weights: inverse-distance blend of the 8 nearest posed vertices, top 4
    d, idx = knn(xc, v_c, 8)
    wd = 1.0 / np.maximum(d, 2e-3) ** 2
    wd /= wd.sum(1, keepdims=True)
    W = np.einsum("nk,nkj->nj", wd, Wv[idx])
    top = np.argsort(-W, axis=1)[:, :4]
    tw = np.take_along_axis(W, top, 1)
    trunc = 1.0 - tw.sum(1)
    tw /= tw.sum(1, keepdims=True)
    joints = top.astype(np.uint16)

    # 3. un-pose to the shaped zero pose
    Mc = np.einsum("nk,nkab->nab", tw, Sc[joints])
    Minv = np.linalg.inv(Mc)
    x0 = np.einsum("nab,nb->na", Minv[:, :3, :3], xc) + Minv[:, :3, 3]
    q0 = quat_mul(quat_conj(mat_to_quat(Mc[:, :3, :3])), qc)
    q0 /= np.linalg.norm(q0, axis=1, keepdims=True)
    # canonical pose round trip (format check 7: posing the canonical pose returns the canonical means)
    xr, _ = lbs(Sc, joints, tw, x0)
    roundtrip_mm = float(np.abs(xr - xc).max() * 1e3)

    checks = {"canonical_pose_fit_median_mm": fits, "canonical_pose_used": pose_name,
              "roundtrip_cano_max_mm": round(roundtrip_mm, 4),
              "weight_truncation_median": round(float(np.median(trunc)), 4),
              "weight_truncation_p99": round(float(np.percentile(trunc, 99)), 4)}

    # 4. versus LHM++'s own posed scene at pose key `key`
    if scene is not None:
        P = np.load(REALISM / "input" / "poses.npz")
        fp = np.zeros((N_J, 3))
        fp[0] = P[f"{role}_root"][key]
        fp[1:22] = P[f"{role}_body"][key].reshape(21, 3)
        fp[22] = P[f"{role}_jaw"][key]
        fp[23] = P[f"{role}_leye"][key]
        fp[24] = P[f"{role}_reye"][key]
        fp[25:40] = P[f"{role}_lhand"][key].reshape(15, 3)
        fp[40:55] = P[f"{role}_rhand"][key].reshape(15, 3)
        trans = P[f"{role}_pelvis"][key] - J[0]
        St = model.skin_mats(model.fk(J, fp, trans), J)
        xp, Mt = lbs(St, joints, tw, x0)
        ref = np.stack([scene["x"], scene["y"], scene["z"]], 1)[scene_offset:scene_offset + n]
        e = np.linalg.norm(xp - ref, axis=1) * 1e3
        qp = quat_mul(mat_to_quat(Mt[:, :3, :3]), q0)
        qref = np.stack([scene["rot_1"], scene["rot_2"], scene["rot_3"], scene["rot_0"]], 1)[scene_offset:scene_offset + n]
        qref /= np.linalg.norm(qref, axis=1, keepdims=True)
        ang = np.degrees(2 * np.arccos(np.clip(np.abs((qp * qref).sum(1)), 0, 1)))
        hairish = d[:, 0] > 0.03
        checks["vs_lhmpp_posed_scene"] = {
            "scene": f"scene_key{key}.ply", "key": key, "t": float(P["t"][key]),
            "centre_err_mm": {"median": round(float(np.median(e)), 2), "p95": round(float(np.percentile(e, 95)), 2),
                              "p99": round(float(np.percentile(e, 99)), 2), "max": round(float(e.max()), 1)},
            "centre_err_mm_far_from_body_gt_3cm": {"count": int(hairish.sum()),
                                                   "median": round(float(np.median(e[hairish])), 2) if hairish.any() else None},
            "rotation_err_deg": {"median": round(float(np.median(ang)), 2), "p95": round(float(np.percentile(ang, 95)), 2)}}

    # 5. Unity coordinates
    rec = np.zeros(n, RECORD_DTYPE)
    rec["mean"] = x0 * MIRROR
    rec["rotation"] = q0 * np.array([-1.0, -1.0, 1.0, 1.0])
    rec["log_scale"] = log_scale
    rec["opacity"] = opac
    rec["sh0"] = sh0
    rec["joints"] = joints
    rec["weights"] = tw
    rec["hair"] = 0
    rec["strand"] = -1
    rec["segment"] = -1
    rec["local_rotation"] = np.array([0.0, 0.0, 0.0, 1.0])
    Ju = J * MIRROR
    cjw = np.tile(np.eye(4), (N_J, 1, 1))
    cjw[:, :3, 3] = Ju
    alpha = 1 / (1 + np.exp(-opac))
    header = {
        "format": "dancecap splat avatar", "version": 1, "role": role, "name": None,
        "capture": "08_JohnLula", "take": "john_lula", "count": n, "hair_count": 0,
        "record": RECORD_TABLE, "coords": "unity (x, y, -z) of the Y-up right-handed dancecap world", "units": "m",
        "colour": "sh0 DC; colour = 0.5 + 0.28209479 * sh0; sRGB-encoded",
        "canonical_pose": {"name": "smplx_zero_pose", "axis_angle": np.zeros((N_J, 3)).tolist()},
        "lhmpp_neutral_pose": {"name": f"lhmpp_cano_ply ({pose_name})", "axis_angle": pose_c.round(6).tolist(),
                               "note": "the pose the source *_cano.ply was found in (step 1); the means were un-posed from it"},
        "betas": betas.round(6).tolist(), "body_model": f"SMPL-X neutral ({model.path.name}, local copy), flat hands",
        "rest_joints": Ju.round(6).tolist(), "parents": model.parents.tolist(),
        "canonical_joint_world": cjw.reshape(N_J, 16).round(6).tolist(),
        "skinning": {"joints_per_gaussian": 4,
                     "source": "APPROXIMATE (jl_splatav): inverse-distance blend of the 8 nearest posed SMPL-X vertices' "
                               "LBS weights, top 4 renormalised; LHM++'s own diffused voxel weights were not exported for john_lula",
                     "truncation": {"median": checks["weight_truncation_median"], "p99": checks["weight_truncation_p99"]}},
        "posing": "SPLAT_AVATAR_FORMAT.md section 4",
        "hair": {"selection": "none (hair = 0 everywhere; john_lula proof only)", "binding": "unbound"},
        "provenance": {"lhmpp_commit": pod.get("commit"), "lhmpp_model": pod.get("model"),
                       "weights_licence": "CC BY-NC 4.0", "smplx_licence": "SMPL-X non-commercial",
                       "inputs": [f"work/john_lula/realism/lhmpp/pod_out/gs/{info['cano_ply']}"],
                       "pod_session": json.loads((REALISM / "report.json").read_text()).get("session", {}).get("session"),
                       "tool": "dancecap.mono4d.jl_splatav", "created": time.strftime("%Y-%m-%dT%H:%M:%S")},
        "checks": checks,
        "stats": {"opacity_p5_50_95": np.percentile(alpha, [5, 50, 95]).round(4).tolist(),
                  "sigma_max_mm_p5_50_95": (np.percentile(np.exp(log_scale).max(1), [5, 50, 95]) * 1e3).round(2).tolist(),
                  "bounds_min": (x0 * MIRROR).min(0).round(4).tolist(), "bounds_max": (x0 * MIRROR).max(0).round(4).tolist()},
        "likeness": "likeness data: work/ and git-ignored StreamingAssets only; never commit",
    }
    h = write_splatav(out_dir, role, rec, header)
    report[role] = {"checks": checks, "count": n, "seconds": round(time.time() - t0, 1), "bytes": h["data_bytes"]}
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--key", type=int, default=3020, help="pose key of the pod's scene_key<k>.ply to check against")
    a = ap.parse_args(argv)
    out = Path(a.out)
    model = Smplx()
    scene_path = REALISM / "pod_out" / "gs" / f"scene_key{a.key}.ply"
    scene = read_ply(scene_path) if scene_path.exists() else None
    report: dict = {}
    off = 0
    for role in ROLES:
        h = convert(role, out, a.key, model, report, scene, off)
        off += h["count"]
    rep = REALISM.parent / "unity_splats" / "jl_splatav_report.json"  # never inside the capture's splats/ folder (the viewer reads every role json there)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
