"""Synthetic take in the dancecap contract format, with ground truth, for testing downstream stages.

Two SMPL-X bodies (neutral model, real FK) doing a zouk-ish basic at 88 BPM in the Y-up world frame:
alternating steps that land exactly on the beat, the couple slowly rotating with an elastic partner gap,
the follow's head/torso circling (head movement), and the follow stepping 90 ms LATE in measures 3-4 of
every 8 (eased in/out over one swing, see _late_shift). Ground truth (beats, every touchdown time per foot, which
steps are late) is written next to it.

    python -m dancecap synthetic           -> work/synthetic/smplx/{lead,follow}.npz, beats.json, truth.json, audio.wav

window() cuts a short (e.g. 6 s, like the real take) slice of the written take, optionally with Gaussian joint
noise and/or SMPL-X parameter noise, for the timing / physics validations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .take import PACKAGE_ROOT, WORKSPACE

BPM = 88.0
BEAT = 60.0 / BPM
MEASURE = 4 * BEAT  # one zouk "measure" = 4 beats = 8 eighths, as in beat_this_analyzer.py
LATE_S = 0.090


def _set(pose, joint, axis, angle):
    pose[:, (joint - 1) * 3 + axis] += angle


LATE_MEASURES = (2, 3)  # measure index mod 8 in which the follow steps late
RAMP = True  # False = the original step-function shift (re-landing / one-frame drop artefacts at its edges)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _late_shift(t: np.ndarray, role: str, ramp: bool | None = None) -> np.ndarray:
    """Time warp of the follow's step cycle: tt = t - shift(t). shift = LATE_S in measures 2-3 (mod 8).

    ramp=True (default) eases the shift in during the swing that lands on the first late beat (the swing is
    stretched by LATE_S) and out during the swing that lands on the first on-time beat (compressed), so every
    touchdown is a normal soft landing and the motion is C1. The step function (ramp=False) made the foot land,
    lift 1.4 cm and re-land at the start of a late block and drop to the floor within one frame at its end."""
    t = np.asarray(t, float)
    if role != "follow":
        return np.zeros_like(t)
    ramp = RAMP if ramp is None else ramp
    if not ramp:
        m = np.floor(t / MEASURE).astype(int) % 8
        return np.where(np.isin(m, LATE_MEASURES), LATE_S, 0.0)
    a, b = LATE_MEASURES[0] * MEASURE, (LATE_MEASURES[-1] + 1) * MEASURE  # late block [a, b) of each 8-measure cycle
    u = t % (8 * MEASURE)
    up = _smoothstep((u - (a - BEAT)) / BEAT)                              # over the swing landing at a
    down = _smoothstep((u - (b - BEAT + LATE_S)) / (BEAT - LATE_S))       # over the swing landing at b
    return LATE_S * np.where(u < a, up, 1.0 - down)


def body_pose(t: np.ndarray, role: str) -> np.ndarray:
    pose = np.zeros((len(t), 63))
    tt = t - _late_shift(t, role)
    phase = (tt / BEAT) % 2.0
    lift_l = np.clip(np.sin(np.pi * phase), 0, None) * (phase < 1)
    lift_r = np.clip(np.sin(np.pi * (phase - 1)), 0, None) * (phase >= 1)
    for hip, knee, ankle, lift in ((1, 4, 7, lift_l), (2, 5, 8, lift_r)):
        _set(pose, hip, 0, -0.45 * lift)
        _set(pose, knee, 0, 0.9 * lift)
        _set(pose, ankle, 0, -0.25 * lift)  # heel comes up during the swing
    _set(pose, 3, 2, 0.06 * np.sin(np.pi * tt / BEAT))
    if role == "follow":
        a = 2 * np.pi * t / (2 * MEASURE)
        for j, g in ((6, 0.12), (9, 0.18), (12, 0.2), (15, 0.25)):
            _set(pose, j, 0, g * np.sin(a))
            _set(pose, j, 2, g * np.cos(a))
    _set(pose, 16, 1, -0.9)
    _set(pose, 16, 2, -0.9)
    _set(pose, 18, 1, -0.9)  # left elbow flexion (+y would hyperextend the arm backwards)
    _set(pose, 17, 1, 1.1)
    _set(pose, 17, 2, 0.5)
    _set(pose, 19, 1, 1.2)
    return pose


def touchdowns(duration: float, role: str) -> list[dict]:
    """Analytic touchdown times: a foot lands when its warped phase tt = t - shift(t) reaches a beat (every other
    beat per foot); t - shift(t) is strictly increasing, so the landing time is its inverse at the beat."""
    tf = np.arange(0.0, duration + 2.0, 1e-4)
    warp = tf - _late_shift(tf, role)
    out = []
    k = 1
    while k * BEAT < duration + 1:
        nominal = k * BEAT
        foot = "left" if k % 2 == 1 else "right"  # left lifts during [2n, 2n+1) beats, lands at odd beats
        t = float(np.interp(nominal, warp, tf))
        late = round(t - nominal, 6)
        late = 0.0 if abs(late) < 1e-5 else late
        if t < duration:
            out.append({"t": round(t, 6), "foot": foot, "nominal_beat_t": round(nominal, 6), "late_s": late})
        k += 1
    return out


def make(out_dir: Path, seconds: float = 24.0, fps: float = 30.0, smplx_model: Path | None = None) -> Path:
    import smplx
    import torch

    model_path = smplx_model or WORKSPACE / "smplx" / "SMPLX_NEUTRAL.npz"
    t = np.arange(int(seconds * fps)) / fps
    yaw = 2 * np.pi * t / (4 * MEASURE)  # couple rotates once every 4 measures
    half_gap = 0.5 * (0.62 + 0.12 * np.sin(2 * np.pi * t / MEASURE))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smplx").mkdir(exist_ok=True)
    model = smplx.create(str(model_path), model_type="smplx", gender="neutral", use_pca=False,
                         flat_hand_mean=True, num_betas=10, batch_size=len(t))
    betas = {"lead": np.array([0.8, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]), "follow": np.array([-0.6, 0.2, 0, 0, 0, 0, 0, 0, 0, 0.0])}

    for role, sign in (("lead", -1.0), ("follow", 1.0)):
        # lead sits at +z (yaw 0) and must face -z towards the follow; canonical SMPL-X faces +z
        facing = yaw + (np.pi if role == "lead" else 0.0)
        go = np.stack([np.zeros_like(t), facing, np.zeros_like(t)], 1)  # yaw about +Y; canonical faces +Z
        bp = body_pose(t, role)
        b = torch.tensor(np.repeat(betas[role][None], len(t), 0), dtype=torch.float32)
        with torch.no_grad():
            rest = model(betas=b[:1].expand(len(t), -1), body_pose=torch.zeros(len(t), 63),
                         global_orient=torch.zeros(len(t), 3))
        rest_j = rest.joints[0].numpy()
        foot_y = rest_j[[10, 11], 1].min()
        pelvis = rest_j[0]
        # partner positions on a circle; each faces the couple centre
        target_xz = np.stack([-sign * half_gap * np.sin(yaw), -sign * half_gap * np.cos(yaw)], 1)
        transl = np.zeros((len(t), 3))
        transl[:, 0] = target_xz[:, 0] - pelvis[0]
        transl[:, 2] = target_xz[:, 1] - pelvis[2]
        transl[:, 1] = -foot_y
        with torch.no_grad():
            out = model(betas=b, body_pose=torch.tensor(bp, dtype=torch.float32),
                        global_orient=torch.tensor(go, dtype=torch.float32),
                        transl=torch.tensor(transl, dtype=torch.float32))
        joints = out.joints.numpy()[:, :55].astype(np.float64)
        np.savez_compressed(
            out_dir / "smplx" / f"{role}.npz", t=t, global_orient=go, body_pose=bp,
            left_hand_pose=np.zeros((len(t), 45)), right_hand_pose=np.zeros((len(t), 45)), jaw_pose=np.zeros((len(t), 3)),
            transl=transl, betas=betas[role], joints=joints, gender="neutral",
            quality=np.ones(len(t)))

    from atlas_bridge.beats import zouk_grid  # noqa: E402  (shared beat format)
    beats = zouk_grid(BPM, seconds + 1.0)
    (out_dir / "beats.json").write_text(json.dumps(beats, indent=1))
    truth = {
        "bpm": BPM, "measure_s": MEASURE, "late_s": LATE_S, "fps": fps, "duration": seconds,
        "late_measures_mod8": [2, 3],
        "touchdowns": {r: touchdowns(seconds, r) for r in ("lead", "follow")},
        "late_ramp": bool(RAMP),
        "notes": "lead steps land exactly on every beat; follow lands 90 ms late in measures 3-4 of every 8 "
                 "(0-based measure index 2,3 mod 8), the shift eased in/out over one swing. World: Y-up, metres, "
                 "floor y=0.",
    }
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=1))
    from atlas_bridge.demo import _write_groove  # noqa: E402
    import atlas_bridge.demo as d
    d.BPM = BPM
    _write_groove(out_dir / "audio.wav", seconds + 1.0)
    return out_dir


def window(work: Path | None = None, t0: float = 0.0, t1: float = 6.0, joint_noise_m: float = 0.0,
           transl_noise_m: float = 0.0, rot_noise_deg: float = 0.0, seed: int = 0) -> dict:
    """Short slice [t0, t1) of a written synthetic take (default work/synthetic), optionally noisy.

    Returns {"npz": {role: dict of arrays (contract keys)}, "truth": truth.json with the touchdowns inside the
    window, "beats": the full [[t, type]] grid (unchanged, so measure indices stay those of the take)}.
    joint_noise_m: iid Gaussian noise on `joints` only (timing reads joints); transl_noise_m / rot_noise_deg:
    noise on transl and every body axis-angle (physics reads the parameters; `joints` is then dropped as stale)."""
    work = Path(work) if work is not None else PACKAGE_ROOT / "work" / "synthetic"
    rng = np.random.default_rng(seed)
    truth = json.loads((work / "truth.json").read_text())
    out = {"npz": {}, "beats": json.loads((work / "beats.json").read_text()), "t0": t0, "t1": t1}
    for role in ("lead", "follow"):
        d = np.load(work / "smplx" / f"{role}.npz")
        t = np.asarray(d["t"], float)
        k = (t >= t0) & (t < t1)
        r = {key: (d[key][k] if d[key].ndim and len(d[key]) == len(t) else d[key]) for key in d.files}
        if joint_noise_m > 0:
            r["joints"] = r["joints"] + rng.normal(0, joint_noise_m, r["joints"].shape)
        if transl_noise_m > 0 or rot_noise_deg > 0:
            from scipy.spatial.transform import Rotation
            n = int(k.sum())
            r["transl"] = r["transl"] + rng.normal(0, transl_noise_m, (n, 3))
            for key, nj in (("global_orient", 1), ("body_pose", 21)):
                rv = np.asarray(r[key], float).reshape(n * nj, 3)
                noise = Rotation.from_rotvec(rng.normal(0, math.radians(rot_noise_deg), (n * nj, 3)))
                r[key] = (noise * Rotation.from_rotvec(rv)).as_rotvec().reshape(n, 3 * nj)
            r.pop("joints", None)
        out["npz"][role] = r
    truth = dict(truth)
    truth["touchdowns"] = {r: [e for e in truth["touchdowns"][r] if t0 <= e["t"] < t1] for r in truth["touchdowns"]}
    truth["window"] = [t0, t1]
    out["truth"] = truth
    return out


def main():
    import sys
    sys.path.insert(0, str(WORKSPACE / "atlas_bridge"))
    out = make(PACKAGE_ROOT / "work" / "synthetic")
    print(f"[synthetic] -> {out}")
