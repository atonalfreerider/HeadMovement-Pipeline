r"""Quest-only DISTILLED splat dancers (2026-10-21; owner after the first Quest 3 test with splats: "framerate was extremely low
but splat looked correct. try deep compression"). A compact Gaussian set per dancer, fitted to renders of the full LHM++ avatar
(160,000 Gaussians), so the headset draws far fewer and smaller Gaussians; the PC files stay untouched.

    python -m dancecap.mono4d.quest_distill prepare [--lessons recap,john_lula] [--sizes 40000,25000,15000]   (local)
                                                    [--roles lead,follow] [--work work/quest_distill]
        -> <work>/bundle/ (the full avatars + sampled poses) + <work>/pod_job.json
    python -m dancecap remote run --jobs <work>/pod_job.json --max-cost <cap> --gpus "..."                    (the pod)
        runs: python -m dancecap.mono4d.quest_distill distill --bundle ... --out ...
    python -m dancecap.mono4d.quest_distill import [--sid <session>] [--work work/quest_distill]               (local)
        -> head-movement/Assets/StreamingAssets/<capture>/splats/quest_<N>/<role>.{splatav,json} (git-ignored)
    --work: a folder of its own per run (e.g. work/quest_distill_ca for carlos_aline), so a new lesson does not upload the
    other lessons' bundle; --roles: only these dancers (e.g. the follow again after her hair binding).

Method (per dancer, per target count N; gsplat 1.5.3 on the pod, `--renderer torch` = a dense pure-torch rasteriser for a
local smoke test only):
  * teacher = the full avatar of the capture (`splats/<role>.splatav`, SPLAT_AVATAR_FORMAT.md), canonical zero pose, plus the
    same avatar posed by its own 4-joint LBS (section 4) with poses sampled from the capture's motion (`smplx_<role>.bin`,
    pelvis held at the rest joint: pose only).
  * student init by importance: score = opacity x sqrt(sigma_mid sigma_max) x (1 + 4 |colour - mean colour of 8 neighbours|),
    N drawn without replacement with p ~ score; attributes copied, scales widened to the student's neighbour spacing.
  * optimise means / log-scales / rotations / colours / opacity logits (Adam) against teacher renders of ~100 canonical views
    (fibonacci sphere, 1.2-3.4 m, plus head / upper-body close-ups) and posed views (24 training poses), random backgrounds:
    0.8 L1 + 0.2 (1 - SSIM) on colour + 0.5 L1 on alpha + a max-scale penalty (sigma above `cap`) + a FILL term: the
    Quest renderer's quad area per Gaussian (opacity-aware bound at alpha 1/255, net.kleinbeck.gaussians: 8 min(ln(a/c), 4)
    sqrt(det Sigma_2D) px^2) summed over the view, relative to the teacher's, so overdraw actually drops.
  * dead Gaussians (opacity < 0.02) are relocated next to live ones every 300 steps (first 70 % of the steps).
  * LBS weights are re-transferred from the 6 nearest teacher Gaussians (inverse-distance, top 4) every 250 steps and at the
    end; hair flags from the nearest teacher Gaussian; a hair Gaussian whose nearest teacher Gaussian is bound to a strand gets
    that strand / segment, its local offset / rotation recomputed from its own centre and rotation in the rest segment frame
    (format section 6, recap_splat_hair.segment_frames).
  * the strand look (2026-10-23 fix round 2): when the teacher's hair carries recap_splat_hair's look (json hair.look.params:
    needle-shaped, stack-capped bound hair, e.g. Aline's black hair), the students that sit on bound hair keep it too -
    while fitting (every step: per-Gaussian opacity <= the stack cap, across sigmas <= needle_across_m, along <=
    needle_along_max_m; every 100 steps and at the end: recap_splat_hair.needle_shape along the bound segment's rest
    tangent, and at the end stack_cap) and in the written set (bound_look) - otherwise the fit re-grows them into large
    opaque Gaussians (480 / 310 / 200 in the 40k / 25k / 15k sets of fix round 1), the black discs a swinging strand
    carries. Teachers without a look (the recap, every lead) are fitted exactly as before.
  * scores on held-out views (16 canonical + 8 held-out poses x 2): PSNR / SSIM / LPIPS (alex, when installed) on the dancer
    crop over grey, against the teacher; the same for the load-time thinning of the viewer (SplatAvatarData.Reduce rule:
    the N largest opacity x (sigma_mid sigma_max)^0.25) at the same N; fill ratio vs the teacher at a Quest-like eye.

Likeness data: work/ and the git-ignored StreamingAssets only; the pod is our own (owner consent 2026-10-19).
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

WORK = PACKAGE_ROOT / "work" / "quest_distill"
BUNDLE = WORK / "bundle"
POD_OUT = WORK / "pod_out"
SA = WORKSPACE / "head-movement" / "Assets" / "StreamingAssets"
LESSONS = {"recap": "06_LarissaKaduRecap", "john_lula": "08_JohnLula", "carlos_aline": "07_CarlosAline",
           "gui_julissa": "09_GuiJulissa"}
ROLES = ("lead", "follow")
SIZES = (40000, 25000, 15000)
SH_C0 = 0.28209479177387814
ALPHA_CUT = 1.0 / 255.0
RECORD = np.dtype([("mean", "<f4", 3), ("rotation", "<f4", 4), ("log_scale", "<f4", 3), ("opacity_logit", "<f4"),
                   ("sh0", "<f4", 3), ("joints", "<u2", 4), ("weights", "<f4", 4), ("hair", "u1"), ("reserved", "u1", 3),
                   ("strand", "<i4"), ("segment", "<i4"), ("local_offset", "<f4", 3), ("local_rotation", "<f4", 4)])
assert RECORD.itemsize == 120
HEAD = 15
FORWARD_UNITY = np.array([0.0, 0.0, -1.0])
UP_UNITY = np.array([0.0, 1.0, 0.0])


def log(msg: str) -> None:
    print(f"[quest_distill {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ================================================================================================ files
def read_splatav(p: Path) -> np.ndarray:
    raw = Path(p).read_bytes()
    assert raw[:4] == b"SPAV", p
    hdr = np.frombuffer(raw[4:64], "<u4")
    version, n, rb, sh_rest, flags, hb = (int(x) for x in hdr[:6])
    assert version == 1 and rb == 120 and hb == 64 and flags & 1 and sh_rest == 0, (version, rb, hb, flags, sh_rest)
    return np.frombuffer(raw[hb:hb + n * rb], RECORD).copy()


def write_splatav(p: Path, rec: np.ndarray) -> None:
    hdr = np.zeros(16, "<u4")
    hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6] = 1, len(rec), 120, 0, 1, 64
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        f.write(b"SPAV")
        f.write(hdr[1:].tobytes())
        f.write(np.ascontiguousarray(rec, RECORD).tobytes())


def read_motion(p: Path) -> tuple[dict, np.ndarray]:
    raw = Path(p).read_bytes()
    n = int(np.frombuffer(raw[4:8], "<u4")[0])
    h = json.loads(raw[8:8 + n])
    return h, np.frombuffer(raw[8 + n:], "<f4").reshape(h["frames"], h["frame_floats"])


# ================================================================================================ quaternions (numpy, xyzw)
def q_mul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a, float), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def q_to_mat(q):
    q = np.asarray(q, float)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def mat_to_q(R):
    R = np.asarray(R, float)
    shp = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    q = np.empty((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    k = np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], 1).argmax(1)
    for i in range(4):
        m = k == i
        if not m.any():
            continue
        r = R[m]
        if i == 0:
            s = np.sqrt(np.maximum(1 + tr[m], 1e-12)) * 2
            q[m] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s, (r[:, 1, 0] - r[:, 0, 1]) / s, 0.25 * s], 1)
        elif i == 1:
            s = np.sqrt(np.maximum(1 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-12)) * 2
            q[m] = np.stack([0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s, (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 2, 1] - r[:, 1, 2]) / s], 1)
        elif i == 2:
            s = np.sqrt(np.maximum(1 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-12)) * 2
            q[m] = np.stack([(r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s, (r[:, 1, 2] + r[:, 2, 1]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s], 1)
        else:
            s = np.sqrt(np.maximum(1 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-12)) * 2
            q[m] = np.stack([(r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s, 0.25 * s, (r[:, 1, 0] - r[:, 0, 1]) / s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q.reshape(*shp, 4)


def fk_skin(rest: np.ndarray, parents: np.ndarray, quats_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Skin matrices of one pose (format section 4, pelvis held at its rest joint): rotations (55,3,3), translations (55,3)."""
    Rl = q_to_mat(quats_xyzw)
    Rw = np.zeros((55, 3, 3))
    Pw = np.zeros((55, 3))
    for j in range(55):
        p = parents[j]
        if p < 0:
            Rw[j], Pw[j] = Rl[j], rest[j]
        else:
            Rw[j] = Rw[p] @ Rl[j]
            Pw[j] = Pw[p] + Rw[p] @ (rest[j] - rest[p])
    t = Pw - np.einsum("jab,jb->ja", Rw, rest)
    return Rw, t


# ================================================================================================ prepare (local)
def _work(work: Path | str | None) -> Path:
    w = Path(work) if work else WORK
    return w if w.is_absolute() else PACKAGE_ROOT / w


def prepare(lessons: list[str], sizes: list[int], n_poses: int = 32, iters: int = 4000, work: Path | str | None = None,
            roles: tuple = ROLES) -> Path:
    work = _work(work)
    BUNDLE = work / "bundle"
    BUNDLE.mkdir(parents=True, exist_ok=True)
    man = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "sizes": sizes, "iters": iters, "dancers": []}
    for lesson in lessons:
        cap = LESSONS[lesson]
        for role in roles:
            src = SA / cap / "splats"
            js, sv = src / f"{role}.json", src / f"{role}.splatav"
            if not js.exists():
                log(f"skip {cap} {role}: no {js}")
                continue
            d = BUNDLE / cap
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(js, d / js.name)
            shutil.copy2(sv, d / sv.name)
            h, fr = read_motion(SA / cap / f"smplx_{role}.bin")
            T = len(fr)
            idx = np.linspace(int(T * 0.04), int(T * 0.96), n_poses).round().astype(int)
            q = fr[idx, 3:3 + 55 * 4].reshape(n_poses, 55, 4)
            ok = np.isfinite(q).all((1, 2)) & (np.abs(np.linalg.norm(q, axis=-1) - 1) < 1e-2).all(1)
            np.save(d / f"poses_{role}.npy", q[ok].astype(np.float32))
            man["dancers"].append({"capture": cap, "lesson": lesson, "role": role, "json": f"{cap}/{js.name}",
                                   "splatav": f"{cap}/{sv.name}", "splatav_sha256": sha256(sv), "poses": f"{cap}/poses_{role}.npy",
                                   "pose_frames": idx[ok].tolist(), "motion_frames": T})
            log(f"{cap} {role}: {sv.stat().st_size / 1e6:.1f} MB, {int(ok.sum())} poses")
    (BUNDLE / "manifest.json").write_text(json.dumps(man, indent=1))
    job = job_spec(work, iters)
    p = work / "pod_job.json"
    p.write_text(json.dumps(job, indent=1))
    log(f"bundle {BUNDLE} ({sum(f.stat().st_size for f in BUNDLE.rglob('*') if f.is_file()) / 1e6:.0f} MB), job {p}")
    return p


def job_spec(work: Path | str | None = None, iters: int = 4000) -> dict:
    """The distill pod job of a run folder (work/quest_distill by default; any folder under dancecap/)."""
    work = _work(work)
    relw = work.resolve().relative_to(PACKAGE_ROOT.resolve()).as_posix()      # e.g. work/quest_distill
    rel = f"dancecap/{relw}"
    return {
        "schema": "dancecap.pod_job/1", "name": "quest_distill" if work == WORK else f"quest_distill_{work.name}",
        "take": "quest_distill",
        "max_minutes": 50, "needs": ["gsplat"],
        "inputs": [f"{rel}/bundle/"],
        "exclude": ["dancecap/work/runpod/pywheels", "mamma-turing"],   # the wheelhouse goes up: the cached gsplat wheel
        "env": {"PYTHONUNBUFFERED": "1"},
        "setup": ["pip install -q --no-deps lpips==0.1.4 || echo 'lpips unavailable: LPIPS is skipped'"],
        "commands": [{"name": "distill", "timeout_min": 45,
                      "cmd": f"python -m dancecap.mono4d.quest_distill distill --bundle {relw}/bundle "
                             f"--out {relw}/pod_out --iters {iters} --time-budget-min 40"}],
        "outputs": [f"{rel}/pod_out/"],
        "verify": [f"{rel}/pod_out/distill_report.json"],
        "after_download": ["python -m dancecap.mono4d.quest_distill import"
                           + ("" if work == WORK else f" --work {relw}")],
        "notes": ["Quest-only distilled splat dancers: compact Gaussian sets fitted to renders of the full LHM++ avatars "
                  "(canonical + posed views), max-scale and fill penalties; scores vs the full model on held-out views",
                  "likeness data goes to our own pod only (owner consent 2026-10-19); outputs stay in work/ and the "
                  "git-ignored StreamingAssets"],
    }


# ================================================================================================ torch side
def _torch():
    import torch
    return torch


class Dancer:
    """One avatar on the GPU: canonical attributes (Unity coords used as a right-handed world: everything is mirrored
    consistently, so renders are mirror images of the real dancer - irrelevant for the fit) + its LBS + its poses."""

    def __init__(self, rec: np.ndarray, header: dict, poses: np.ndarray, dev):
        torch = _torch()
        self.rec, self.header, self.dev = rec, header, dev
        self.rest = np.asarray(header["rest_joints"], float)
        self.parents = np.asarray(header["parents"], int)
        f = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
        self.mu = f(rec["mean"])
        q = rec["rotation"].astype(np.float64)
        self.q = f(np.concatenate([q[:, 3:], q[:, :3]], 1))  # wxyz
        self.s = f(np.exp(rec["log_scale"]))
        self.o = f(1 / (1 + np.exp(-rec["opacity_logit"].astype(np.float64))))
        self.rgb = f(np.clip(0.5 + SH_C0 * rec["sh0"], 0, 1))
        self.j = torch.as_tensor(rec["joints"].astype(np.int64), device=dev)
        self.w = f(rec["weights"])
        self.poses = poses
        self._skin_cache: dict[int, tuple] = {}
        o = rec["opacity_logit"]
        op = 1 / (1 + np.exp(-o.astype(np.float64)))
        self.centre = (rec["mean"] * op[:, None]).sum(0) / op.sum()
        self.lo, self.hi = np.percentile(rec["mean"], 1, 0), np.percentile(rec["mean"], 99, 0)

    def skin(self, k: int):
        """(R (55,3,3), t (55,3), q (55,4) wxyz) torch tensors of pose k (cached)."""
        torch = _torch()
        if k not in self._skin_cache:
            R, t = fk_skin(self.rest, self.parents, self.poses[k])
            qx = mat_to_q(R)
            qw = np.concatenate([qx[:, 3:], qx[:, :3]], 1)
            f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=self.dev)
            self._skin_cache[k] = (f(R), f(t), f(qw))
        return self._skin_cache[k]

    def posed_centre(self, k: int) -> np.ndarray:
        R, t = fk_skin(self.rest, self.parents, self.poses[k])
        Pw = np.einsum("jab,jb->ja", R, self.rest) + t
        return Pw[[0, 3, 6, 9, 12, 15]].mean(0)


def q_mul_t(a, b):
    """wxyz Hamilton product (torch, broadcasting)."""
    torch = _torch()
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def lbs(mu, q, joints, weights, skin):
    """Posed centres and rotations (format section 4; rotation by the sign-aligned weighted joint quaternion, the shader-
    friendly rule of the format doc)."""
    torch = _torch()
    R, t, qj = skin
    Rk = R[joints]                                   # (N,4,3,3)
    A = (weights[..., None, None] * Rk).sum(1)
    b = (weights[..., None] * t[joints]).sum(1)
    mu_p = torch.einsum("nab,nb->na", A, mu) + b
    qk = qj[joints]                                  # (N,4,4)
    sgn = torch.sign((qk * qk[:, :1]).sum(-1, keepdim=True))
    sgn = torch.where(sgn == 0, torch.ones_like(sgn), sgn)
    qb = (weights[..., None] * sgn * qk).sum(1)
    qb = qb / qb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return mu_p, q_mul_t(qb, q)


def look_at(cam: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = target - cam
    f /= np.linalg.norm(f)
    up = np.array([0.0, 1.0, 0.0]) if abs(f[1]) < 0.97 else np.array([0.0, 0.0, 1.0])
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    Rm = np.stack([r, d, f])
    V = np.eye(4)
    V[:3, :3] = Rm
    V[:3, 3] = -Rm @ cam
    return V


def fib_dirs(n: int, seed: int, el_lo: float = -25, el_hi: float = 70) -> np.ndarray:
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    lo, hi = math.sin(math.radians(el_lo)), math.sin(math.radians(el_hi))
    y = lo + (hi - lo) * (i / n)
    az = math.pi * (1 + 5 ** 0.5) * i + rng.uniform(0, 2 * math.pi)
    r = np.sqrt(1 - y * y)
    return np.stack([r * np.cos(az), y, r * np.sin(az)], 1)


def view_set(d: Dancer, n: int, seed: int, posed: list[int] | None = None, size: int = 512) -> list[dict]:
    """Canonical (posed=None) or posed views: body views 1.2-3.4 m (vertical fov fitting a ~2.1 m tall frame at 3 m) and
    close-ups of the head (0.5-0.9 m) / upper body (0.9-1.4 m)."""
    rng = np.random.default_rng(seed)
    dirs = fib_dirs(n, seed)
    out = []
    head = d.rest[HEAD]
    for i in range(n):
        k = None if posed is None else posed[i % len(posed)]
        centre = d.centre if k is None else d.posed_centre(k)
        kind = rng.choice(["body", "body", "body", "upper", "head"])
        if kind == "body":
            dist, tgt = rng.uniform(1.2, 3.4), centre + rng.normal(0, 0.08, 3)
            fov = math.degrees(2 * math.atan(1.15 / dist)) if dist < 3.0 else 40.0
        elif kind == "upper":
            dist = rng.uniform(0.9, 1.4)
            tgt = (centre + (head - d.rest[0]) * 0.6) + rng.normal(0, 0.05, 3)
            fov = 50.0
        else:
            dist = rng.uniform(0.5, 0.9)
            hpos = head if k is None else (lambda R, t: R[HEAD] @ d.rest[HEAD] + t[HEAD])(*fk_skin(d.rest, d.parents, d.poses[k]))
            tgt = hpos + np.array([0, 0.03, 0]) + rng.normal(0, 0.03, 3)
            fov = 45.0
        cam = tgt + dirs[i] * dist
        fy = 0.5 * size / math.tan(math.radians(fov) / 2)
        K = np.array([[fy, 0, size / 2], [0, fy, size / 2], [0, 0, 1.0]])
        out.append({"V": look_at(cam, tgt), "K": K, "pose": k, "kind": kind, "dist": float(dist)})
    return out


# ------------------------------------------------------------------------------------------------ renderers
def render_gsplat(means, quats, scales, opac, cols, Vs, Ks, W, H, bg):
    torch = _torch()
    from gsplat import rasterization
    img, alpha, meta = rasterization(means, quats, scales, opac, cols, Vs, Ks, W, H, near_plane=0.05, far_plane=100.0,
                                     packed=False, backgrounds=bg, rasterize_mode="classic")
    return img, alpha, meta


def project_cov(means, quats, scales, Vs, Ks, eps2d=0.3):
    """Torch reference of the EWA projection: means2d (C,N,2), depth (C,N), conic (C,N,3), det(cov2d) (C,N)."""
    torch = _torch()
    q = quats / quats.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    Rq = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                      torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                      torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    M = Rq * scales[:, None, :]
    cov = M @ M.transpose(-1, -2)
    Rv, tv = Vs[:, :3, :3], Vs[:, :3, 3]
    pc = torch.einsum("cab,nb->cna", Rv, means) + tv[:, None]
    zc = pc[..., 2].clamp_min(1e-4)
    fx, fy, cx, cy = Ks[:, 0, 0, None], Ks[:, 1, 1, None], Ks[:, 0, 2, None], Ks[:, 1, 2, None]
    u = fx * pc[..., 0] / zc + cx
    v = fy * pc[..., 1] / zc + cy
    J = torch.zeros(pc.shape[:2] + (2, 3), device=means.device)
    J[..., 0, 0] = fx / zc
    J[..., 0, 2] = -fx * pc[..., 0] / zc ** 2
    J[..., 1, 1] = fy / zc
    J[..., 1, 2] = -fy * pc[..., 1] / zc ** 2
    covc = torch.einsum("cab,nbd,ced->cnae", Rv, cov, Rv)
    c2 = J @ covc @ J.transpose(-1, -2)
    a, b, c = c2[..., 0, 0] + eps2d, c2[..., 0, 1], c2[..., 1, 1] + eps2d
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], -1)
    return torch.stack([u, v], -1), pc[..., 2], conic, det


def render_torch(means, quats, scales, opac, cols, Vs, Ks, W, H, bg):
    """Dense pure-torch rasteriser (smoke tests: small N, small images): sorted front-to-back alpha compositing."""
    torch = _torch()
    m2, depth, conic, det = project_cov(means, quats, scales, Vs, Ks)
    ys, xs = torch.meshgrid(torch.arange(H, device=means.device) + 0.5, torch.arange(W, device=means.device) + 0.5, indexing="ij")
    imgs, alphas = [], []
    for ci in range(Vs.shape[0]):
        ok = (depth[ci] > 0.05) & (m2[ci, :, 0].abs() < 4 * W) & (m2[ci, :, 1].abs() < 4 * H)
        idx = ok.nonzero().flatten()
        order = idx[torch.argsort(depth[ci, idx])]
        dx = xs.reshape(-1, 1) - m2[ci, order, 0][None]
        dy = ys.reshape(-1, 1) - m2[ci, order, 1][None]
        cc = conic[ci, order]
        pw = (-0.5 * (cc[:, 0] * dx * dx + 2 * cc[:, 1] * dx * dy + cc[:, 2] * dy * dy)).clamp(max=0.0)
        al = (opac[order][None] * torch.exp(pw)).clamp(max=0.99)
        al = torch.where(al < ALPHA_CUT, torch.zeros_like(al), al)
        T = torch.cumprod(torch.cat([torch.ones_like(al[:, :1]), 1 - al[:, :-1]], 1), 1)
        wgt = al * T
        img = wgt @ cols[order] + (1 - wgt.sum(1, keepdim=True)) * bg[ci][None]
        imgs.append(img.reshape(H, W, 3))
        alphas.append(wgt.sum(1).reshape(H, W, 1))
    vis = (depth > 0.05) & (m2[..., 0].abs() < 4 * W) & (m2[..., 1].abs() < 4 * H)
    meta = {"conics": conic, "radii": vis.float()[..., None].expand(-1, -1, 2) * 3, "depths": depth}
    return torch.stack(imgs), torch.stack(alphas), meta


def fill_px(meta, opac, W, H) -> "torch.Tensor":
    """The Quest renderer's fragment count per view (net.kleinbeck.gaussians: quad half extent = sqrt(ln(a / cutoff)) capped at
    2 in exp(-r^2) units = 8 min(ln(a/c), 4) sqrt(det Sigma_2D) px^2 per Gaussian), summed over the Gaussians that project
    in front of the camera; (C,) tensor, differentiable in the conics and opacities."""
    torch = _torch()
    con = meta["conics"]
    detc = (con[..., 0] * con[..., 2] - con[..., 1] ** 2).clamp_min(1e-12)
    vis = (meta["radii"][..., 0] > 0) if meta["radii"].dim() == 3 else (meta["radii"] > 0)
    lnr = torch.log(opac.clamp_min(1e-6) / ALPHA_CUT).clamp(0, 4)
    area = 8 * lnr[None] * detc.rsqrt()
    area = torch.where(vis, area, torch.zeros_like(area)).clamp(max=float(W * H))
    return area.sum(-1)


def ssim(x, y):
    """x, y (C,H,W,3) in 0..1 -> mean SSIM (11 px gaussian window)."""
    torch = _torch()
    import torch.nn.functional as F
    x, y = x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)
    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None]).expand(3, 1, 11, 11).contiguous()
    mu_x, mu_y = F.conv2d(x, k, padding=5, groups=3), F.conv2d(y, k, padding=5, groups=3)
    sxx = F.conv2d(x * x, k, padding=5, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, k, padding=5, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, k, padding=5, groups=3) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))).mean()


# ------------------------------------------------------------------------------------------------ student
def knn_teacher(query, ref, k: int, chunk: int = 1024):
    torch = _torch()
    ds, ids = [], []
    for i0 in range(0, len(query), chunk):
        dd = torch.cdist(query[i0:i0 + chunk], ref)
        d_, i_ = dd.topk(k, dim=1, largest=False)
        ds.append(d_)
        ids.append(i_)
    return torch.cat(ds), torch.cat(ids)


def transfer_weights(mu, d: Dancer, k: int = 6):
    """4-joint LBS weights of student centres from the k nearest teacher Gaussians (inverse distance, top 4, renormalised)."""
    torch = _torch()
    dist, idx = knn_teacher(mu.detach(), d.mu, k)
    iw = 1.0 / (dist + 1e-4)
    iw = iw / iw.sum(1, keepdim=True)
    full = torch.zeros(len(mu), 55, device=mu.device)
    jj = d.j[idx]                                  # (n,k,4)
    ww = d.w[idx] * iw[..., None]                  # (n,k,4)
    full.scatter_add_(1, jj.reshape(len(mu), -1), ww.reshape(len(mu), -1))
    w4, j4 = full.topk(4, dim=1)
    w4 = w4.clamp_min(0)
    w4 = w4 / w4.sum(1, keepdim=True).clamp_min(1e-9)
    return j4, w4, idx[:, 0], dist[:, 0]


def look_params(header: dict) -> dict | None:
    """The teacher's strand look (recap_splat_hair.apply_look params in json hair.look.params), or None."""
    hs = header.get("hair")
    lk = ((hs.get("look") or {}).get("params")) if isinstance(hs, dict) else None
    return lk if isinstance(lk, dict) and "needle_across_m" in lk and "stack_cap" in lk else None


def rest_guides(header: dict) -> np.ndarray | None:
    """The Unity hair guides' rest points of a bound file (json hair.binding_info.rest_points), (guides, points, 3)."""
    hs = header.get("hair")
    hb = hs.get("binding_info") if isinstance(hs, dict) else None
    if not hb or "rest_points" not in hb:
        return None
    return np.asarray(hb["rest_points"], float).reshape(int(hb["guides"]), int(hb["points"]), 3)


def _large_opaque(op: np.ndarray, S: np.ndarray, idx: np.ndarray) -> int:
    return int(((S.max(1) > 0.008) & (op > 0.5))[idx].sum())


def bound_look(rec: np.ndarray, header: dict) -> tuple[np.ndarray, dict]:
    """The teacher's strand look on a set's bound records (strand >= 0; in place): recap_splat_hair.needle_shape along the
    rest tangent of each record's segment (local_rotation re-expressed in the segment frame) and stack_cap on the opacity.
    No look in the header or nothing bound: unchanged. -> (rec, info)"""
    from . import recap_splat_hair as RH
    lk, rest = look_params(header), rest_guides(header)
    b = np.nonzero(rec["strand"] >= 0)[0]
    if lk is None or rest is None or not len(b):
        return rec, {"applied": False}
    op = 1 / (1 + np.exp(-rec["opacity_logit"].astype(np.float64)))
    before = _large_opaque(op, np.exp(rec["log_scale"].astype(np.float64)), b)
    s_, k_ = rec["strand"][b].astype(int), rec["segment"][b].astype(int)
    q = rec["rotation"][b].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q2, ls2 = RH.needle_shape(q, rec["log_scale"][b], rest[s_, k_ + 1] - rest[s_, k_], lk["needle_across_m"],
                              lk["needle_along_max_m"])
    F = RH.segment_frames(rest)[s_, k_]
    lrot = RH.q_mul(RH.q_conj(RH.q_from_mat(F)), q2)
    rec["rotation"][b] = q2
    rec["log_scale"][b] = ls2
    rec["local_rotation"][b] = lrot / np.linalg.norm(lrot, axis=1, keepdims=True)
    o2, ci = RH.stack_cap(rec["mean"][b].astype(np.float64), op[b], lk["stack_cap"], lk["stack_link_m"])
    o2 = np.clip(o2, 1e-6, 1 - 1e-6)
    ch = ~np.isclose(o2, op[b], rtol=0, atol=1e-9)
    rec["opacity_logit"][b[ch]] = np.log(o2[ch] / (1 - o2[ch]))
    op3 = 1 / (1 + np.exp(-rec["opacity_logit"].astype(np.float64)))
    S3 = np.exp(rec["log_scale"].astype(np.float64))
    return rec, {"applied": True, "bound": int(len(b)), "stack_cap": ci,
                 "large_opaque_bound": {"before": before, "after": _large_opaque(op3, S3, b),
                                        "rule": "sigma_max > 8 mm and opacity > 0.5"},
                 "opacity_max": round(float(op3[b].max()), 4), "across_mm_max": round(float(S3[b].min(1).max()) * 1e3, 3)}


def look_project(lk: dict, sb, log_s, logit, q=None, tangents=None) -> None:
    """In-training projection of the students on bound hair (sb: bool mask tensor): opacity <= the stack cap, the two
    smaller sigmas <= across, the largest <= along max; with q (wxyz) / tangents (numpy, one per masked student) also the
    needle orientation (recap_splat_hair.needle_shape). In place on .data."""
    torch = _torch()
    if not bool(sb.any()):
        return
    with torch.no_grad():
        ii = sb.nonzero().flatten()
        logit.data[ii] = logit.data[ii].clamp(max=math.log(lk["stack_cap"] / (1 - lk["stack_cap"])))
        ls = log_s.data[ii]
        srt, order = ls.sort(dim=1)
        hi = torch.tensor([math.log(lk["needle_across_m"]), math.log(lk["needle_across_m"]),
                           math.log(lk["needle_along_max_m"])], dtype=ls.dtype, device=ls.device)
        log_s.data[ii] = torch.empty_like(ls).scatter_(1, order, torch.minimum(srt, hi))
        if q is not None and tangents is not None:
            from . import recap_splat_hair as RH
            qw = q.data[ii].cpu().numpy().astype(np.float64)
            qx = np.concatenate([qw[:, 1:], qw[:, :1]], 1)
            q2, ls2 = RH.needle_shape(qx, log_s.data[ii].cpu().numpy().astype(np.float64), tangents,
                                      lk["needle_across_m"], lk["needle_along_max_m"])
            q.data[ii] = torch.as_tensor(np.concatenate([q2[:, 3:], q2[:, :3]], 1), dtype=q.dtype, device=q.device)
            log_s.data[ii] = torch.as_tensor(ls2, dtype=log_s.dtype, device=log_s.device)


def init_student(d: Dancer, n: int, seed: int):
    torch = _torch()
    g = torch.Generator(device="cpu").manual_seed(seed)
    s_sorted = d.s.sort(dim=1).values
    area = (s_sorted[:, 1] * s_sorted[:, 2]).sqrt()
    _, nn = knn_teacher(d.mu, d.mu, 9)
    detail = (d.rgb - d.rgb[nn[:, 1:]].mean(1)).abs().sum(1)
    score = (d.o * area * (1 + 4 * detail)).clamp_min(1e-12)
    sel = torch.multinomial(score.cpu(), n, replacement=False, generator=g).to(d.mu.device)
    mu = d.mu[sel].clone()
    dist, _ = knn_teacher(mu, mu, 4)
    spacing = dist[:, 1:].mean(1, keepdim=True)
    s = torch.sqrt(d.s[sel] ** 2 + (0.45 * spacing) ** 2)
    o = d.o[sel].clamp(0.3, 0.95)
    return {"mu": mu, "q": d.q[sel].clone(), "log_s": s.log(), "logit": torch.log(o / (1 - o)), "rgb": d.rgb[sel].clone(),
            "sel": sel}


def prune_baseline(d: Dancer, n: int):
    """The viewer's load-time thinning (SplatAvatarData.Reduce): the n largest opacity x (sigma_mid sigma_max)^0.25."""
    s_sorted = d.s.sort(dim=1).values
    score = d.o * (s_sorted[:, 1] * s_sorted[:, 2]) ** 0.25
    return score.topk(n).indices


# ------------------------------------------------------------------------------------------------ distill one dancer
def distill_one(d: Dancer, n: int, iters: int, renderer, args, seed: int = 0) -> dict:
    torch = _torch()
    dev = d.dev
    rng = np.random.default_rng(seed + n)
    P = len(d.poses)
    train_p = [k for k in range(P) if k % 4 != 1]
    test_p = [k for k in range(P) if k % 4 == 1]
    size = args.size
    canon_v = view_set(d, 100, 11, None, size)
    posed_v = {k: view_set(d, 6, 1000 + k, [k], size) for k in train_p}
    st = init_student(d, n, seed)
    mu = st["mu"].requires_grad_(True)
    q = st["q"].requires_grad_(True)
    log_s = st["log_s"].requires_grad_(True)
    logit = st["logit"].requires_grad_(True)
    rgb = st["rgb"].requires_grad_(True)
    ext = float(np.linalg.norm(d.hi - d.lo))
    opt = torch.optim.Adam([{"params": [mu], "lr": 1.6e-4 * ext, "name": "mu"}, {"params": [q], "lr": 1e-3, "name": "q"},
                            {"params": [log_s], "lr": 5e-3, "name": "s"}, {"params": [logit], "lr": 3e-2, "name": "o"},
                            {"params": [rgb], "lr": 2.5e-3, "name": "c"}], eps=1e-15)
    j4, w4, near, _ = transfer_weights(mu, d)
    cap = args.cap_m if args.cap_m > 0 else {40000: 0.012, 25000: 0.016, 15000: 0.02}.get(n, 0.02)
    lk, rest_g = look_params(d.header), rest_guides(d.header)
    tb_np = (d.rec["strand"] >= 0) & (d.rec["hair"] != 0)
    if lk is not None and rest_g is not None and tb_np.any():      # the teacher's strand look (module doc)
        tan_np = np.zeros((len(d.rec), 3))
        bi = np.nonzero(tb_np)[0]
        s_t, k_t = d.rec["strand"][bi].astype(int), d.rec["segment"][bi].astype(int)
        tan_np[bi] = rest_g[s_t, k_t + 1] - rest_g[s_t, k_t]
        tb = torch.as_tensor(tb_np, device=dev)
        sb = tb[near]
        log(f"  N={n}: strand look on {int(sb.sum())} students over bound hair (cap {lk['stack_cap']}, across "
            f"{lk['needle_across_m'] * 1e3:.1f} mm)")
    else:
        lk = None
    t0 = time.time()
    hist = []
    B = args.batch
    for it in range(iters):
        if args.deadline and time.time() > args.deadline:
            log(f"  time budget reached at step {it}")
            break
        frac = it / max(1, iters - 1)
        opt.param_groups[0]["lr"] = 1.6e-4 * ext * (0.01 ** frac)
        posed = rng.random() < 0.45
        if posed:   # one pose per batch: its own views (aimed at that pose's body / head)
            k = train_p[int(rng.integers(0, len(train_p)))]
            vs = [posed_v[k][i] for i in rng.integers(0, len(posed_v[k]), B)]
        else:
            k = None
            vs = [canon_v[i] for i in rng.integers(0, len(canon_v), B)]
        Vs = torch.as_tensor(np.stack([v["V"] for v in vs]), dtype=torch.float32, device=dev)
        Ks = torch.as_tensor(np.stack([v["K"] for v in vs]), dtype=torch.float32, device=dev)
        bg = torch.as_tensor(rng.random((B, 3)), dtype=torch.float32, device=dev)
        with torch.no_grad():
            if posed:
                tmu, tq = lbs(d.mu, d.q, d.j, d.w, d.skin(k))
            else:
                tmu, tq = d.mu, d.q
            timg, talpha, tmeta = renderer(tmu, tq, d.s, d.o, d.rgb, Vs, Ks, size, size, bg)
            tfill = fill_px(tmeta, d.o, size, size)
        smu, sq = (lbs(mu, q, j4, w4, d.skin(k)) if posed else (mu, q))
        so = torch.sigmoid(logit)
        img, alpha, meta = renderer(smu, sq, log_s.exp(), so, rgb.clamp(0, 1), Vs, Ks, size, size, bg)
        l1 = (img - timg).abs().mean()
        lss = 1 - ssim(img, timg)
        la = (alpha - talpha).abs().mean()
        fill = fill_px(meta, so, size, size)
        fill_ratio = (fill / tfill.clamp_min(1.0)).mean()
        smax = log_s.exp().max(1).values
        lcap = (torch.relu(smax - cap) / cap).pow(2).mean()
        loss = 0.8 * l1 + 0.2 * lss + 0.5 * la + args.w_fill * fill_ratio + args.w_cap * lcap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.data = q.data / q.data.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            log_s.data.clamp_(math.log(2e-4), math.log(0.06))
        if it % 250 == 249:
            j4, w4, near, _ = transfer_weights(mu, d)
            if lk is not None:
                sb = tb[near]
        if lk is not None:
            orient = it % 100 == 99
            look_project(lk, sb, log_s, logit, q if orient else None,
                         tan_np[near[sb].cpu().numpy()] if orient else None)
        if it % 300 == 299 and frac < 0.7:
            with torch.no_grad():
                so = torch.sigmoid(logit)
                dead = (so < 0.02).nonzero().flatten()
                if len(dead):
                    alive = (so >= 0.02).nonzero().flatten()
                    src = alive[torch.multinomial(so[alive], len(dead), replacement=True)]
                    sc = log_s.exp()[src]
                    mu.data[dead] = mu.data[src] + torch.randn_like(mu.data[src]) * sc * 0.5
                    log_s.data[dead] = (sc * 0.7).log()
                    log_s.data[src] = (sc * 0.85).log()
                    q.data[dead] = q.data[src]
                    rgb.data[dead] = rgb.data[src]
                    o_new = so[src].clamp(0.05, 0.9)
                    logit.data[dead] = torch.log(o_new / (1 - o_new))
                    for group in opt.param_groups:
                        for p in group["params"]:
                            stt = opt.state.get(p)
                            if stt:
                                stt["exp_avg"][dead] = 0
                                stt["exp_avg_sq"][dead] = 0
                    j4[dead], w4[dead] = j4[src], w4[src]
                hist.append({"step": it + 1, "relocated": int(len(dead))})
        if it % 200 == 0 or it == iters - 1:
            hist.append({"step": it + 1, "loss": round(float(loss), 5), "l1": round(float(l1), 5), "alpha_l1": round(float(la), 5),
                         "fill_ratio": round(float(fill_ratio), 4), "smax_p99_mm": round(float(smax.quantile(0.99)) * 1e3, 2),
                         "s": round(time.time() - t0, 1)})
            if it % 1000 == 0 or it == iters - 1:
                log(f"  N={n} step {it + 1}/{iters}: {hist[-1]}")
    j4, w4, near, dnear = transfer_weights(mu, d)
    look_info = None
    if lk is not None:   # the written set's look (bound_look) on the tensors too, so the scores below are the file's
        sb = tb[near]
        look_project(lk, sb, log_s, logit, q, tan_np[near[sb].cpu().numpy()])
        if bool(sb.any()):
            from . import recap_splat_hair as RH
            ii = sb.nonzero().flatten()
            o_np = torch.sigmoid(logit.data[ii]).cpu().numpy().astype(np.float64)
            o2, ci = RH.stack_cap(mu.data[ii].cpu().numpy().astype(np.float64), o_np, lk["stack_cap"], lk["stack_link_m"])
            o2 = np.clip(o2, 1e-6, 1 - 1e-6)
            logit.data[ii] = torch.as_tensor(np.log(o2 / (1 - o2)), dtype=logit.dtype, device=logit.device)
            look_info = {"students_on_bound_hair": int(sb.sum()), "stack_cap": ci}
    out = {k_: v.detach() for k_, v in {"mu": mu, "q": q, "log_s": log_s, "logit": logit, "rgb": rgb}.items()}
    out.update(j4=j4, w4=w4, near=near, dnear=dnear, hist=hist, steps=it + 1, seconds=round(time.time() - t0, 1), cap=cap,
               test_poses=test_p, look=look_info)
    return out


# ------------------------------------------------------------------------------------------------ evaluation
def evaluate(d: Dancer, students: dict, renderer, args, lp=None, sheet_dir: Path | None = None, tag: str = "") -> dict:
    """students: name -> (mu, q, s, o, rgb, j4, w4). Held-out canonical views + held-out poses; crop to the dancers."""
    torch = _torch()
    dev, size = d.dev, args.size
    P = len(d.poses)
    test_p = [k for k in range(P) if k % 4 == 1]
    views = view_set(d, 16, 101, None, size) + view_set(d, 2 * len(test_p), 102, sorted(test_p * 2), size)
    res = {name: {"psnr": [], "ssim": [], "lpips": [], "fill_ratio": []} for name in students}
    bgv = torch.full((1, 3), 0.5, device=dev)
    sheets = []
    for vi, v in enumerate(views):
        Vs = torch.as_tensor(v["V"][None], dtype=torch.float32, device=dev)
        Ks = torch.as_tensor(v["K"][None], dtype=torch.float32, device=dev)
        with torch.no_grad():
            k = v["pose"]
            tmu, tq = lbs(d.mu, d.q, d.j, d.w, d.skin(k)) if k is not None else (d.mu, d.q)
            timg, talpha, tmeta = renderer(tmu, tq, d.s, d.o, d.rgb, Vs, Ks, size, size, bgv)
            tf = float(fill_px(tmeta, d.o, size, size)[0])
            m = talpha[0, ..., 0] > 0.02
            ys, xs = torch.nonzero(m, as_tuple=True)
            if len(ys) == 0:
                continue
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            row = [timg[0]]
            for name, (smu, sq, ss, so, sc, j4, w4) in students.items():
                pmu, pq = lbs(smu, sq, j4, w4, d.skin(k)) if k is not None else (smu, sq)
                img, alpha, meta = renderer(pmu, pq, ss, so, sc, Vs, Ks, size, size, bgv)
                a, b = img[0, y0:y1, x0:x1], timg[0, y0:y1, x0:x1]
                mse = ((a - b) ** 2).mean().clamp_min(1e-10)
                res[name]["psnr"].append(float(-10 * torch.log10(mse)))
                res[name]["ssim"].append(float(ssim(a[None], b[None])))
                if lp is not None and (y1 - y0) >= 32 and (x1 - x0) >= 32:
                    res[name]["lpips"].append(float(lp(a.permute(2, 0, 1)[None] * 2 - 1, b.permute(2, 0, 1)[None] * 2 - 1)))
                res[name]["fill_ratio"].append(float(fill_px(meta, so, size, size)[0]) / max(tf, 1.0))
                row.append(img[0])
            if sheet_dir is not None and vi in (0, 3, 6, 9, 16, 18, 20) and len(sheets) < 7:
                # each image mirrored back to the real handedness (Unity coordinates are a z-mirror), columns kept in order
                sheets.append(torch.cat([r_.flip(1) for r_ in row], 1)[y0:y1].clamp(0, 1).cpu().numpy())
    summ = {}
    for name, r in res.items():
        summ[name] = {kk: (round(float(np.mean(vv)), 4) if vv else None) for kk, vv in r.items()}
        summ[name]["psnr_min"] = round(float(np.min(r["psnr"])), 3) if r["psnr"] else None
        summ[name]["views"] = len(r["psnr"])
    if sheet_dir is not None and sheets:
        from PIL import Image
        W = max(s.shape[1] for s in sheets)
        rows = [np.pad(s, ((0, 4), (0, W - s.shape[1]), (0, 0)), constant_values=1.0) for s in sheets]
        im = (np.concatenate(rows, 0) * 255).round().astype(np.uint8)
        sheet_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(im).save(sheet_dir / f"{tag}.jpg", quality=88)
        summ["sheet"] = f"{tag}.jpg"
        summ["sheet_columns"] = "(mirrored back to the real handedness) teacher 160k | " + " | ".join(students)
    return summ


# ------------------------------------------------------------------------------------------------ export
def to_records(d: Dancer, st: dict) -> tuple[np.ndarray, dict]:
    from . import recap_splat_hair as RH
    mu = st["mu"].cpu().numpy().astype(np.float64)
    qw = st["q"].cpu().numpy().astype(np.float64)
    qw /= np.linalg.norm(qw, axis=1, keepdims=True)
    qx = np.concatenate([qw[:, 1:], qw[:, :1]], 1)
    n = len(mu)
    rec = np.zeros(n, RECORD)
    rec["mean"] = mu
    rec["rotation"] = qx
    rec["log_scale"] = st["log_s"].cpu().numpy()
    o = 1 / (1 + np.exp(-st["logit"].cpu().numpy().astype(np.float64)))
    o = np.clip(o, 1e-6, 1 - 1e-6)
    rec["opacity_logit"] = np.log(o / (1 - o))
    rec["sh0"] = (np.clip(st["rgb"].cpu().numpy(), 0, 1) - 0.5) / SH_C0
    rec["joints"] = st["j4"].cpu().numpy().astype(np.uint16)
    w = st["w4"].cpu().numpy().astype(np.float64)
    rec["weights"] = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
    near = st["near"].cpu().numpy()
    t = d.rec[near]
    rec["hair"] = t["hair"]
    rec["strand"], rec["segment"] = -1, -1
    rec["local_rotation"] = np.array([0.0, 0.0, 0.0, 1.0])
    info = {"bound": 0, "hair": int((rec["hair"] != 0).sum())}
    hb = (d.header.get("hair") or {}).get("binding_info")
    if (d.header.get("hair") or {}).get("binding") == "unity_hair_strands" and hb:
        G, Pn = int(hb["guides"]), int(hb["points"])
        rest = np.asarray(hb["rest_points"], float).reshape(G, Pn, 3)
        bound = (t["strand"] >= 0) & (rec["hair"] != 0)
        if bound.any():
            s_, g_ = t["strand"][bound].astype(int), t["segment"][bound].astype(int)
            c = mu[bound] - np.asarray(d.header["rest_joints"], float)[HEAD]
            F = RH.segment_frames(rest)[s_, g_]
            off = np.einsum("nji,nj->ni", F, c - rest[s_, g_])
            lrot = RH.q_mul(RH.q_conj(RH.q_from_mat(F)), qx[bound])
            rec["strand"][bound], rec["segment"][bound] = s_, g_
            rec["local_offset"][bound] = off
            rec["local_rotation"][bound] = lrot / np.linalg.norm(lrot, axis=1, keepdims=True)
            info["bound"] = int(bound.sum())
            info["distance_to_teacher_mm_p50_p90"] = [round(float(np.percentile(st["dnear"].cpu().numpy()[bound], q_)) * 1e3, 2) for q_ in (50, 90)]
            if look_params(d.header) is not None:
                rec, info["look"] = bound_look(rec, d.header)
                if st.get("look"):
                    info["look"]["fit"] = st["look"]
    return rec, info


def recount_hair(h: dict, rec: np.ndarray) -> dict:
    """A distilled set's json header copies the teacher's (the full avatar's) hair section: rewrite its counts from the SET's
    own records (hair_count, hair.hair_count, hair.bound, hair.binding_info.bound); the teacher's numbers are kept as
    hair.teacher_hair_count / hair.teacher_bound. -> h (changed in place)"""
    hair_n = int((rec["hair"] != 0).sum())
    bound_n = int((rec["strand"] >= 0).sum())
    h["hair_count"] = hair_n
    hs = h.get("hair")
    if isinstance(hs, dict):
        if "teacher_hair_count" not in hs and "hair_count" in hs:
            hs["teacher_hair_count"] = hs["hair_count"]
        if "teacher_bound" not in hs and "bound" in hs:
            hs["teacher_bound"] = hs["bound"]
        if "hair_count" in hs:
            hs["hair_count"] = hair_n
        if "bound" in hs or bound_n:
            hs["bound"] = bound_n
        if isinstance(hs.get("binding_info"), dict):
            hs["binding_info"]["bound"] = bound_n
    return h


def write_student(dest: Path, d: Dancer, rec: np.ndarray, extra: dict) -> dict:
    role = d.header["role"]
    p = dest / f"{role}.splatav"
    write_splatav(p, rec)
    h = json.loads(json.dumps(d.header))
    sc = np.exp(rec["log_scale"].astype(np.float64))
    op = 1 / (1 + np.exp(-rec["opacity_logit"].astype(np.float64)))
    h.update({"data": p.name, "data_bytes": p.stat().st_size, "data_sha256": sha256(p), "count": int(len(rec)),
              "hair_count": int((rec["hair"] != 0).sum())})
    h["stats"] = {"opacity_median": round(float(np.median(op)), 4), "opacity_below_0.05": int((op < 0.05).sum()),
                  "scale_max_mm_p50_p90_max": [round(float(np.percentile(sc.max(1), q_) * 1e3), 2) for q_ in (50, 90, 100)],
                  "bounds_min": rec["mean"].min(0).round(4).tolist(), "bounds_max": rec["mean"].max(0).round(4).tolist()}
    if "hair" in h and isinstance(h["hair"], dict) and h["hair"].get("binding") == "unity_hair_strands":
        h["hair"] = dict(h["hair"])
        bi = dict(h["hair"].get("binding_info") or {})
        bi["bound"] = extra.get("hair", {}).get("bound")
        bi["distilled_note"] = "strand / segment of the nearest bound teacher Gaussian; local offset / rotation recomputed"
        h["hair"]["binding_info"] = bi
    recount_hair(h, rec)
    h["quest_distill"] = extra
    (dest / f"{role}.json").write_text(json.dumps(h, indent=1))
    return h


# ------------------------------------------------------------------------------------------------ distill (pod)
def distill(args) -> dict:
    torch = _torch()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer = render_torch if args.renderer == "torch" else render_gsplat
    bundle, out = Path(args.bundle), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((bundle / "manifest.json").read_text())
    sizes = [int(s) for s in (args.sizes.split(",") if args.sizes else man["sizes"])]
    lp = None
    if not args.no_lpips:
        try:
            import lpips
            lp = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
        except Exception as e:  # noqa: BLE001
            log(f"LPIPS unavailable ({e}): skipped")
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "device": str(dev), "renderer": args.renderer,
              "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else None, "iters": args.iters, "size": args.size,
              "batch": args.batch, "w_fill": args.w_fill, "w_cap": args.w_cap, "dancers": []}
    try:
        import gsplat
        report["gsplat"] = gsplat.__version__
    except Exception:  # noqa: BLE001
        pass
    t_all = time.time()
    args.deadline = t_all + args.time_budget_min * 60 if args.time_budget_min else None
    todo = [x for x in man["dancers"] if not args.only or f"{x['capture']}/{x['role']}" in args.only.split(",")]
    for di, dn in enumerate(todo):
        rec = read_splatav(bundle / dn["splatav"])
        header = json.loads((bundle / dn["json"]).read_text())
        poses = np.load(bundle / dn["poses"]).astype(np.float64)
        if args.max_teacher:
            keep = np.random.default_rng(0).choice(len(rec), min(args.max_teacher, len(rec)), replace=False)
            rec = rec[np.sort(keep)]
        d = Dancer(rec, header, poses, dev)
        log(f"{dn['capture']} {dn['role']}: teacher {len(rec)} Gaussians, {len(poses)} poses")
        entry = {"capture": dn["capture"], "role": dn["role"], "teacher": len(rec), "sizes": {}}
        students = {}
        for n in sizes:
            if n >= len(rec):
                continue
            left = (args.deadline - time.time()) if args.deadline else None
            remaining = (len(todo) - di) * len(sizes)
            iters = args.iters
            if left is not None:
                # keep inside the budget: share what is left among the remaining fits (eval ~ 10 %)
                per = left / max(1, remaining - list(sizes).index(n)) * 0.85
                iters = int(min(args.iters, max(400, per / max(args.sec_per_iter, 1e-3))))
            t1 = time.time()
            st = distill_one(d, n, iters, renderer, args)
            args.sec_per_iter = max(1e-3, (time.time() - t1) / max(1, st["steps"]))
            ss, so = st["log_s"].exp(), torch.sigmoid(st["logit"])
            students[f"distilled_{n // 1000}k"] = (st["mu"], st["q"], ss, so, st["rgb"].clamp(0, 1), st["j4"], st["w4"])
            sel = prune_baseline(d, n)
            students[f"thinned_{n // 1000}k"] = (d.mu[sel], d.q[sel], d.s[sel], d.o[sel], d.rgb[sel], d.j[sel], d.w[sel])
            rec_s, hinfo = to_records(d, st)
            entry["sizes"][str(n)] = {"steps": st["steps"], "seconds": st["seconds"], "cap_m": st["cap"], "hair": hinfo,
                                      "history": st["hist"]}
            dest = out / dn["capture"] / f"quest_{n // 1000}k"
            entry["sizes"][str(n)]["files"] = str(dest.relative_to(out))
            entry["sizes"][str(n)]["_rec"] = rec_s
        ev = evaluate(d, students, renderer, args, lp, out / "sheets", f"{dn['capture']}_{dn['role']}")
        entry["eval"] = ev
        for n_s, e in entry["sizes"].items():
            n = int(n_s)
            name = f"distilled_{n // 1000}k"
            extra = {"method": "dancecap.mono4d.quest_distill (gsplat fit to the full avatar's renders, canonical + posed views)",
                     "teacher_count": len(rec), "teacher_sha256": dn["splatav_sha256"], "count": n, "steps": e["steps"],
                     "size_px": args.size, "cap_m": e["cap_m"], "w_fill": args.w_fill,
                     "scores_vs_teacher": ev.get(name), "thinned_same_count": ev.get(f"thinned_{n // 1000}k"),
                     "hair": e["hair"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                     "likeness": "photographic likeness of a real person (owner consent 2026-10-19); never commit, never in an APK"}
            write_student(out / e["files"], d, e.pop("_rec"), extra)
        report["dancers"].append(entry)
        (out / "distill_report.json").write_text(json.dumps(report, indent=1))
        del d
        torch.cuda.empty_cache() if dev.type == "cuda" else None
    report["seconds"] = round(time.time() - t_all, 1)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (out / "distill_report.json").write_text(json.dumps(report, indent=1))
    log(f"done in {report['seconds']} s")
    return report


# ================================================================================================ import (local)
def import_run(sid: str | None = None, src: Path | None = None, work: Path | str | None = None) -> dict:
    work = _work(work)
    src = src or work / "pod_out"
    rep = json.loads((src / "distill_report.json").read_text())
    rows = []
    for e in rep["dancers"]:
        for n_s, z in e["sizes"].items():
            s = src / z["files"]
            dest = SA / e["capture"] / "splats" / Path(z["files"]).name
            dest.mkdir(parents=True, exist_ok=True)
            for f in (f"{e['role']}.splatav", f"{e['role']}.json"):
                shutil.copy2(s / f, dest / f)
            h = json.loads((dest / f"{e['role']}.json").read_text())
            assert sha256(dest / f"{e['role']}.splatav") == h["data_sha256"]
            old = json.dumps(h, sort_keys=True)
            recount_hair(h, read_splatav(dest / f"{e['role']}.splatav"))        # the set's own counts (not the teacher's)
            if json.dumps(h, sort_keys=True) != old:
                (dest / f"{e['role']}.json").write_text(json.dumps(h, indent=1))
            sc = e["eval"].get(f"distilled_{int(n_s) // 1000}k", {})
            th = e["eval"].get(f"thinned_{int(n_s) // 1000}k", {})
            rows.append({"capture": e["capture"], "role": e["role"], "n": int(n_s), "psnr": sc.get("psnr"), "ssim": sc.get("ssim"),
                         "lpips": sc.get("lpips"), "fill_ratio": sc.get("fill_ratio"), "thinned_psnr": th.get("psnr"),
                         "thinned_lpips": th.get("lpips"), "thinned_fill_ratio": th.get("fill_ratio"), "dest": str(dest)})
            log(f"{e['capture']} {e['role']} {n_s}: PSNR {sc.get('psnr')} (thinned {th.get('psnr')}), LPIPS {sc.get('lpips')} "
                f"(thinned {th.get('lpips')}), fill {sc.get('fill_ratio')} (thinned {th.get('fill_ratio')}) -> {dest}")
    (work / "import.json").write_text(json.dumps({"session": sid, "rows": rows}, indent=1))
    return {"rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dancecap.mono4d.quest_distill")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--lessons", default="recap,john_lula")
    p.add_argument("--sizes", default=",".join(str(s) for s in SIZES))
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--work", default=None, help="the run's folder (default work/quest_distill)")
    p.add_argument("--roles", default=",".join(ROLES))
    p = sub.add_parser("distill")
    p.add_argument("--bundle", default=str(BUNDLE))
    p.add_argument("--out", default=str(POD_OUT))
    p.add_argument("--sizes", default="")
    p.add_argument("--only", default="")
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--w-fill", dest="w_fill", type=float, default=0.02)
    p.add_argument("--w-cap", dest="w_cap", type=float, default=0.05)
    p.add_argument("--cap-m", dest="cap_m", type=float, default=0.0)
    p.add_argument("--renderer", choices=("gsplat", "torch"), default="gsplat")
    p.add_argument("--max-teacher", dest="max_teacher", type=int, default=0, help="smoke tests: subsample the teacher")
    p.add_argument("--time-budget-min", dest="time_budget_min", type=float, default=0.0)
    p.add_argument("--no-lpips", action="store_true")
    p = sub.add_parser("import")
    p.add_argument("--sid", default=None)
    p.add_argument("--src", default=None)
    p.add_argument("--work", default=None, help="the run's folder (default work/quest_distill)")
    a = ap.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.lessons.split(","), [int(s) for s in a.sizes.split(",")], iters=a.iters, work=a.work,
                roles=tuple(r for r in a.roles.split(",") if r))
    elif a.cmd == "distill":
        a.sec_per_iter = 0.05
        distill(a)
    else:
        import_run(a.sid, Path(a.src) if a.src else None, a.work)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
