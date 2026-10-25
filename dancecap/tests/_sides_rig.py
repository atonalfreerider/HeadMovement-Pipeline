"""Synthetic camera rigs and take folders for the sides tests (no real take data)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

K0 = np.array([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]])


def look_at(C, target=(0.0, 1.0, 0.0)):
    """OpenCV camera at C looking at target (x right, y down): (R, t) with x_cam = R X + t"""
    C = np.asarray(C, float)
    z = np.asarray(target, float) - C
    z /= np.linalg.norm(z)
    down = np.array([0.0, -1.0, 0.0])
    x = np.cross(down, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ C


def ring(n: int, radius: float = 4.0, height: float = 1.6, phase: float = 0.0):
    out = []
    for i in range(n):
        a = phase + 2 * np.pi * i / n
        C = np.array([radius * np.sin(a), height, radius * np.cos(a)])
        R, t = look_at(C)
        out.append((C, R, t))
    return out


def project(R, t, X, K=K0):
    h = (K @ (R @ X + t))
    return h[:2] / h[2]


def stack_rig(rig, T: int):
    """per-view arrays over T frames: P (V,T,3,4), C (V,T,3), f (V,T), and the obs dict of sides.adapters.MultiView"""
    P = np.stack([np.repeat((K0 @ np.hstack([R, t[:, None]]))[None], T, 0) for C, R, t in rig])
    Cc = np.stack([np.repeat(C[None], T, 0) for C, R, t in rig])
    f = np.ones((len(rig), T))
    return P, Cc, f


def body_joints(T: int, cx: float, cz: float, facing: float = 1.0, rng=None) -> np.ndarray:
    """(T,55,3) a standing body (only the 22 body joints + palms matter), pelvis at (cx, 1.0, cz), facing +z * facing"""
    J = np.zeros((T, 55, 3))
    left = np.array([facing, 0, 0.0])
    base = {0: (0, 1.0, 0), 1: (0.1, 0.95, 0), 2: (-0.1, 0.95, 0), 3: (0, 1.1, 0), 4: (0.1, 0.55, 0.02), 5: (-0.1, 0.55, 0.02),
            6: (0, 1.22, 0), 7: (0.1, 0.1, 0), 8: (-0.1, 0.1, 0), 9: (0, 1.35, 0), 10: (0.1, 0.03, 0.12), 11: (-0.1, 0.03, 0.12),
            12: (0, 1.5, 0), 13: (0.07, 1.45, 0), 14: (-0.07, 1.45, 0), 15: (0, 1.62, 0.02), 16: (0.2, 1.45, 0), 17: (-0.2, 1.45, 0),
            18: (0.24, 1.18, 0), 19: (-0.24, 1.18, 0), 20: (0.26, 0.93, 0.05), 21: (-0.26, 0.93, 0.05), 23: (0.03, 1.66, 0.09),
            24: (-0.03, 1.66, 0.09)}
    for j, (x, y, z) in base.items():
        J[:, j] = np.array([cx + x * facing, y, cz + z * facing])
    # MCP joints (palm centre) near the wrists
    for j in range(25, 40):
        J[:, j] = J[:, 20] + np.array([0.0, -0.08, 0.0])
    for j in range(40, 55):
        J[:, j] = J[:, 21] + np.array([0.0, -0.08, 0.0])
    if rng is not None:
        J = J + rng.normal(0, 0.002, J.shape)
    return J


def write_take(root: Path, name: str, cams=("00", "01", "02", "03"), T: int = 60, offsets=(0.1, -0.05), swap: bool = False,
               hz: float = 30.0) -> Path:
    """a stitched take of two chunks (frames 0..T/2-1 owned by chunk 'ca', the rest by 'cb', each with its own video offsets),
    teachers = the exact projections of the base joints (rows swapped when swap=True), phone clocks at 60 fps; returns the work dir"""
    work = root / name
    (work / "people").mkdir(parents=True)
    (work / "cameras").mkdir()
    (work / "smplx_rf2_noslip").mkdir()
    obs_dir = work / "sides" / "obs"
    obs_dir.mkdir(parents=True)
    t = np.arange(T) / hz
    owner = np.where(np.arange(T) < T // 2, "ca", "cb")
    JL = body_joints(T, 0.0, -0.5, 1.0)
    JF = body_joints(T, 0.0, 0.5, -1.0)
    drift = (0.02 * np.arange(T))[:, None]           # 2 cm per frame along x: a wrong video offset shows as a pixel error
    JL[..., 0] += drift
    JF[..., 0] += drift
    for r, J in (("lead", JL), ("follow", JF)):
        np.savez(work / "smplx_rf2_noslip" / f"{r}.npz", t=t, segment_owner=owner, joints=J, betas=np.zeros(10),
                 global_orient=np.zeros((T, 3)), body_pose=np.zeros((T, 63)), transl=np.zeros((T, 3)))
    st = {"chunks": [{"name": "ca", "k": [0, T // 2 + 5]}, {"name": "cb", "k": [T // 2 - 5, T - 1]}],
          "grid": {"frame_offset": {"ca": 0, "cb": T // 2 - 5}, "hz": hz, "T": T}}
    (work / "stitch.json").write_text(json.dumps(st))
    for nm, d in zip(("ca", "cb"), offsets):
        cw = root / f"{name}_{nm}"
        (cw / "mamma").mkdir(parents=True)
        (cw / "sync").mkdir()
        (cw / "mamma" / "select.json").write_text(json.dumps({"cams": list(cams), "chosen": "c"}))
        (cw / "sync" / "offsets.json").write_text(json.dumps({"offsets": {c: {"delta": d} for c in cams}}))
    rig = ring(len(cams))
    from dancecap.sides import geom
    nph = 2 * T + 40
    tph = np.arange(nph) / 60.0 - 0.3                 # the phone's own clock, 60 fps
    for (C, R, tv), c in zip(rig, cams):
        Kc = np.repeat(K0[None], nph, 0)
        np.savez(work / "cameras" / f"{c}.npz", K=Kc, dist=np.zeros(nph), t=tph, R=np.repeat(R[None], nph, 0), tvec=np.repeat(tv[None], nph, 0),
                 ok=np.ones(nph, bool), native_size=np.array([1920, 1080]), pts=tph)
        np.savez(obs_dir / f"cam{c}.npz", K=np.repeat(K0[None], T, 0), R=np.repeat(R[None], T, 0), t=np.repeat(tv[None], T, 0),
                 ok=np.ones(T, bool), vis=np.ones((T, 2, 512), np.float16), size=np.array([1920, 1080]))
        kp = np.full((nph, 2, 17, 3), np.nan, np.float32)
        # phone frame i is shown at grid time tph[i] + delta(owner); store for each phone frame the projection of the base at that time
        for i in range(nph):
            for nm, d in zip(("ca", "cb"), offsets):
                tg = tph[i] + d
                k = int(round(tg * hz))
                if 0 <= k < T and owner[k] == nm and abs(tg - k / hz) < 1 / 50:
                    for p, J in enumerate((JL, JF) if not swap else (JF, JL)):
                        X = J[k][list(geom.C2S)]
                        uv = np.array([project(R, tv, x) for x in X])
                        kp[i, p, :, :2] = uv
                        kp[i, p, :, 2] = 0.9
        np.savez(work / "people" / f"teachers_{c}.npz", t=tph, kp=kp, conf=np.full((nph, 2), 0.95, np.float32))
    return work
