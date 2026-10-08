"""fit_quality: block table / flags / coverage arrays from synthetic fits (pose_quality itself is mocked: no analysis
runs locally)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dancecap import fit_quality as fq


def test_quality_flag_codes():
    assert fq.quality_flag_codes(np.array([0, 1, 2, 3, 4, 7])).tolist() == [0, 1, 2, 3, 3, 3]


def test_block_table_edges_and_stats():
    t = np.arange(-1.0, 21.0, 1 / 30)
    T = len(t)
    obs = np.zeros((2, 2, T), bool)
    obs[0, :, :] = True                                  # camera A sees both roles throughout
    obs[1, 0, t > 12] = True                             # camera B sees the lead from 12 s
    px = {"raw": np.where(obs, 8.0, np.nan).astype(np.float32), "refined": np.where(obs, 6.0, np.nan).astype(np.float32)}
    n_cams = obs.any(1).sum(0).astype(np.int16)
    rows = fq.block_table(t, ["01", "05"], px, obs, n_cams)
    assert [r["t0"] for r in rows] == [-10.0, 0.0, 10.0, 20.0]
    b10 = rows[2]
    assert set(b10["cams"]) == {"01", "05"} and set(b10["cams"]["05"]) == {"lead"}
    assert b10["cams"]["01"]["lead"]["raw"]["median"] == 8.0 and b10["cams"]["01"]["lead"]["refined"]["median"] == 6.0
    assert b10["n_cams_min"] == 1 and b10["n_cams_max"] == 2
    assert rows[0]["cams"]["01"]["lead"]["frames_observed"] == 30   # t = -1.0 .. -1/30; t ~ 0 belongs to the next block


def _fake_work(tmp: Path, T=60, cams=("01", "05")):
    take = "fake_seg"
    w = tmp / "work" / take
    seq = f"{take}_c2"
    (w / "mamma" / "ma_2d" / seq).mkdir(parents=True)
    t = fq.ANCHOR + (np.arange(T) - 998) / fq.HZ
    prep = {"take": take, "grid": {"t": t.tolist(), "T": T}, "sets": {"c2": {"seq": seq, "cams": list(cams)}}}
    (w / "mamma" / "prepare.json").write_text(json.dumps(prep))
    for i, c in enumerate(cams):
        vis = np.zeros((T, 2, 512), np.float32)
        vis[:, 0, :100] = 1.0                            # lead seen by every camera
        if i == 0:
            vis[:, 1, :100] = 1.0                        # follow only by the first
        np.savez(w / "mamma" / "ma_2d" / seq / f"cam{c}.npz", landmarks=np.zeros((T, 2, 512, 3), np.float32),
                 visibilities=vis, contacts=vis, floor_contacts=vis)
    for d in ("smplx", "smplx_rf2_noslip"):
        (w / d).mkdir()
        for r in fq.ROLES:
            np.savez(w / d / f"{r}.npz", quality=np.full(T, 0.7), repaired=np.zeros(T, bool))
    return take, w, t


def test_run_builds_coverage_and_json(monkeypatch, tmp_path):
    from dancecap import pose_quality as pq
    take, w, t = _fake_work(tmp_path)
    monkeypatch.setattr(fq, "PACKAGE_ROOT", tmp_path)
    T = len(t)

    def fake_run(take_, smplx_dir="smplx", tag="final", ref_dir="smplx", **k):
        (w / "quality").mkdir(exist_ok=True)
        base = 5.0 if smplx_dir == "smplx" else 4.0
        fr = {}
        for c in ("01", "05"):
            for r in fq.ROLES:
                a = np.full(T, base)
                if c == "05" and r == "lead":
                    a[10:20] = 40.0                      # a bad stretch in one camera
                fr[f"px_{c}_{r}"] = a
        for r in fq.ROLES:
            fr[f"allpx_{r}"] = np.full(T, base)
        np.savez(w / "quality" / f"quality_{smplx_dir}_frames.npz", **fr)
        return {"reproj_px": {"all_cams_per_frame_median": {"lead": {"median": base}}}, "floor": {}, "slip": {}}

    monkeypatch.setattr(pq, "run", fake_run)
    rep = fq.run(take)
    z = np.load(w / "quality" / "fit_coverage.npz")
    assert z["n_cams"].tolist() == [2] * T and z["n_cams_follow"].tolist() == [1] * T
    assert z["k"][0] == -998 and z["k"][-1] == -998 + T - 1
    assert z["quality_flag_follow"].tolist() == [1] * T and z["quality_flag_lead"].tolist() == [2] * T
    assert z["flag_raw"].shape == (2, 2, T) and z["flag_raw"][1, 0, 10:20].all() and not z["flag_raw"][0].any()
    assert np.isnan(z["px_raw"][1, 1]).all()             # camera 05 does not observe the follow
    assert z["bad_raw"][10:20].all() and z["bad_raw"].sum() == 10   # median of (5, 40) = 22.5 px > 15 on those frames only
    j = json.loads((w / "quality" / "fit_quality.json").read_text())
    assert j["schema"] == "dancecap.fit_quality/1" and set(j["labels"]) == {"raw", "refined"}
    assert j["per_camera"]["05"]["lead"]["raw"]["max"] == 40.0
    assert isinstance(j["per_camera"]["05"]["kind"], dict)
    assert j["blocks"] and j["n_cams"]["no_camera_frames"] == 0
