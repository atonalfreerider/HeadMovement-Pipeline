"""dancecap.mamma_seams on a tiny synthetic chunked fit (no MAMMA, no SMPL-X)."""
import json

import numpy as np

from dancecap import mamma_seams as ms
from dancecap.mamma_chunks import _seam_weights, plan_chunks


def _traj(T):
    t = np.arange(T) / 30.0
    J = np.zeros((T, 55, 3))
    J[:, :, 0] = 0.15 * np.sin(2 * np.pi * 1.0 * t)[:, None]  # dance-like pelvis sway: ~6 m/s^2 peaks
    J[:, :, 2] = 0.3 * np.cos(0.9 * t)[:, None]
    J[:, :, 1] = 0.9 + 0.02 * np.arange(55)[None, :] + 0.05 * np.sin(2.1 * t)[:, None] * (np.arange(55) > 0)
    return t, J


def _write(tmp, take, merged_fn, offset_m=0.02):
    T, n, ov = 150, 3, 24
    chunks = plan_chunks(T, n, ov)
    root = tmp / "work" / take
    (root / "mamma").mkdir(parents=True)
    doc = {"tag": "c8", "T": T, "overlap": ov,
           "chunks": [{"a": a, "b": b, "run_tag": f"c8_f{a}-{b}"} for a, b in chunks]}
    (root / "mamma" / "chunks.json").write_text(json.dumps(doc))
    t, J = _traj(T)
    own = []
    for k, (a, b) in enumerate(chunks):
        Jc = J[a:b].copy()
        Jc[..., 0] += offset_m * k  # each chunk found a slightly shifted solution
        own.append(Jc)
        d = root / "mamma" / "fits" / f"c8_f{a}-{b}" / "smplx"
        d.mkdir(parents=True)
        for role in ("lead", "follow"):
            np.savez(d / f"{role}.npz", t=t[a:b], joints=Jc)
    merged = merged_fn(T, chunks, own)
    (root / "smplx").mkdir()
    for role in ("lead", "follow"):
        np.savez(root / "smplx" / f"{role}.npz", t=t, joints=merged, chunk_ranges=np.array(chunks))
    return root


def _cosine_merge(T, chunks, own):
    out = np.zeros_like(own[0][:1].repeat(T, 0))
    wsum = np.zeros(T)
    for k, ((a, b), Jc) in enumerate(zip(chunks, own)):
        w = np.ones(b - a)
        if k > 0:
            ov = chunks[k - 1][1] - a
            w[:ov] = _seam_weights(ov)
        if k < len(chunks) - 1:
            ov = b - chunks[k + 1][0]
            w[-ov:] = np.minimum(w[-ov:], 1 - _seam_weights(ov))
        out[a:b] += w[:, None, None] * Jc
        wsum[a:b] += w
    return out / wsum[:, None, None]


def _hard_cut(T, chunks, own):
    out = np.zeros_like(own[0][:1].repeat(T, 0))
    for (a, b), Jc in zip(chunks, own):
        out[a:b] = Jc  # later chunk overwrites: a jump at its first frame... inside the overlap, not the window
    for k in range(1, len(chunks)):  # put the jump in the middle of each blend window
        a, b = chunks[k][0], chunks[k - 1][1]
        mid = (a + b) // 2
        out[a:mid] = own[k - 1][a - chunks[k - 1][0]:mid - chunks[k - 1][0]]
    return out


def test_cosine_merge_is_smooth(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PACKAGE_ROOT", tmp_path)
    root = _write(tmp_path, "syn", _cosine_merge, offset_m=0.02)
    res = ms.analyse("syn")
    assert res["merged_is_chunked"]
    for role in ms.ROLES:
        seams = res["roles"][role]["seams"]
        assert len(seams) == 2
        for s in seams:
            assert abs(s["disagree_mm"]["pelvis_median"] - 20.0) < 0.5
            assert s["smooth"], s
    img = ms.sheet(res, root / "review" / "pose_seams.jpg")
    assert img.exists() and img.stat().st_size > 10000


def test_hard_cut_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PACKAGE_ROOT", tmp_path)
    _write(tmp_path, "syn", _hard_cut, offset_m=0.05)
    res = ms.analyse("syn")
    assert not res["all_smooth"]
    s = res["roles"]["lead"]["seams"][0]
    assert s["pelvis_acc"]["ratio_vs_chunks"] > 2
    assert any(x["in_blend_window"] for x in res["roles"]["lead"]["top_pelvis_acc"])
