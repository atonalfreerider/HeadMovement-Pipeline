"""larissa_kadu_full front-end helpers: sync gauge pinning, segment TOML values, identity overlap."""
import tomllib

import numpy as np

from dancecap import front_full as ff
from dancecap.sync import regauge


def _res(deltas):
    return {"offsets": {c: {"delta_ms": d, "delta": d / 1000, "joint_ms": d, "use": True,
                            "sub_windows": [{"t_mid": 1.0, "delta_ms": d + 1}]} for c, d in deltas.items()},
            "curves": {c: {"coarse": [(d, 1.0)], "fine": [(d, 0.5)]} for c, d in deltas.items()},
            "sub_curves": {c: [{"t_mid": 1.0, "fine": [(d, 0.4)]}] for c, d in deltas.items()}}


def test_regauge_pins_median_difference_to_zero():
    ref = {"offsets": {"01": {"delta_ms": 90.0, "use": True}, "02": {"delta_ms": -15.0, "use": True},
                       "03": {"delta_ms": 8.0, "use": True}}}
    res = _res({"01": 70.0, "02": -33.0, "03": -12.0, "07": 5.0})  # same relative offsets, gauge shifted by -20
    rec = regauge(res, ref, "ref")
    assert rec["applied"] and rec["shift_ms"] == 20.0
    o = res["offsets"]
    assert o["01"]["delta_ms"] == 90.0 and o["02"]["delta_ms"] == -13.0 and o["03"]["delta_ms"] == 8.0
    assert o["07"]["delta_ms"] == 25.0 and "delta_vs_ref_ms" not in o["07"]
    assert abs(o["01"]["delta"] - 0.09) < 1e-9 and o["01"]["sub_windows"][0]["delta_ms"] == 91.0
    assert res["curves"]["01"]["fine"][0][0] == 90.0
    assert np.median(list(rec["diff_after_ms"].values())) == 0.0


def test_regauge_without_shared_cameras_is_a_no_op():
    res = _res({"07": 5.0})
    assert regauge(res, {"offsets": {"01": {"delta_ms": 1.0}}}, "ref")["applied"] is False
    assert res["offsets"]["07"]["delta_ms"] == 5.0


def test_toml_values_round_trip():
    d = {"a": 1, "b": 0.875, "c": "Ficar Sem Você \"x\"", "d": [1.0, 0.9], "e": {"sole_heel_m": 0.029}, "f": True}
    txt = "\n".join(f"{k} = {ff._tv(v)}" for k, v in d.items())
    assert tomllib.loads(txt) == d


def test_identity_overlap_detects_a_swap():
    from dancecap.front_full_review import identity_overlap

    class Fake:
        def __init__(self, pts, lead, follow):
            self.cams = ["01"]
            self._p, self._b = np.asarray(pts, float), {"lead": np.asarray(lead, float), "follow": np.asarray(follow, float)}

        def pts(self, c):
            return self._p

        def t(self, c):
            return self._p

        def boxes(self, c, role):
            return self._b[role]
    L, F = [0, 0, 10, 20], [30, 0, 40, 20]
    a = Fake([0.0, 1.0, 2.0], [L, L, L], [F, F, F])
    b = Fake([1.0, 2.0, 3.0], [L, F, L], [F, L, F])  # frame at pts 2.0 swapped
    r = identity_overlap(a, b)["01"]
    assert r["shared_frames"] == 2 and r["lead"]["swapped_frames"] == 1 and not r["passed"]


def test_loc_gate_stats_gap_and_blocks():
    t = np.arange(0, 30, 1 / 30)
    ok = np.ones(len(t), bool)
    ok[(t >= 12) & (t < 21.5)] = False  # 9.5 s failed stretch
    g = ff.loc_gate_stats(t, ok, np.full(len(t), 200), np.full(len(t), 1.0))
    assert abs(g["max_gap_s"] - 9.5) < 0.05 and not g["pass"]
    assert [b["t0"] for b in g["blocks"]] == [0.0, 10.0, 20.0]
    assert g["blocks"][0]["ok_frac"] == 1.0 and g["blocks"][1]["ok_frac"] == 0.2
    ok[(t >= 12) & (t < 21.5)] = True
    ok[::20] = False  # sparse single failures: 95 % ok, tiny gaps
    g = ff.loc_gate_stats(t, ok, np.full(len(t), 200), np.full(len(t), 1.0))
    assert g["pass"] and g["max_gap_s"] < 0.1 and g["strong_frac"] == g["ok_frac"]


def test_pin_betas_file_matches_reference_fit(tmp_path):
    import json
    from dancecap.mamma_dyn import ROLES
    out = ff.pin_betas(tmp_path / "pin.json")
    d = json.loads(out.read_text())
    assert set(d) == {str(k) for k in ROLES}
    for pid, role in ROLES.items():
        ref = np.load(ff.ref_work() / "smplx" / f"{role}.npz")["betas"].reshape(-1)
        assert np.array_equal(np.asarray(d[str(pid)]), ref)


def test_ingest_probe_falls_back_for_old_ffprobe(monkeypatch):
    """The Runpod image's apt ffprobe 4.4 rejects -show_entries stream_side_data=...: probe must use -show_streams."""
    import shutil
    import subprocess
    from pathlib import Path

    import pytest

    from dancecap import ingest
    src = Path(ff.full_cfg()["source_dir"])
    vids = sorted((ff.PACKAGE_ROOT / "takes" / src).resolve().glob("0[56]-Demo-*.mp4"))
    if not vids or not shutil.which("ffprobe"):
        pytest.skip("source videos / ffprobe not available")
    new = {v.name: ingest.probe(v) for v in vids}
    real = subprocess.run

    def old_ffprobe(cmd, *a, **k):
        if "stream_side_data=rotation" in " ".join(map(str, cmd)):
            return subprocess.CompletedProcess(cmd, 1, "", "Failed to set value 'stream_side_data=rotation'")
        return real(cmd, *a, **k)
    monkeypatch.setattr(ingest.subprocess, "run", old_ffprobe)
    old = {v.name: ingest.probe(v) for v in vids}
    assert old == new and all(abs(x["rotation"]) == 90 for x in old.values())


def test_identity_overlap_matches_rounded_pts():
    """Pod-decoded PTS (ffmpeg 4.4: 6 significant digits) still pair with the exact PTS of larissa_kadu_38."""
    from dancecap.front_full_review import identity_overlap

    class Fake:
        def __init__(self, pts, lead, follow):
            self.cams = ["01"]
            self._p, self._b = np.asarray(pts, float), {"lead": np.asarray(lead, float), "follow": np.asarray(follow, float)}

        def pts(self, c):
            return self._p

        def t(self, c):
            return self._p

        def boxes(self, c, role):
            return self._b[role]
    L, F = [0, 0, 10, 20], [30, 0, 40, 20]
    exact = [123.456522, 123.489855, 123.523188]
    rounded = [123.457, 123.49, 123.523]
    r = identity_overlap(Fake(exact, [L] * 3, [F] * 3), Fake(rounded, [L] * 3, [F] * 3))["01"]
    assert r["shared_frames"] == 3 and r["passed"]


def test_showinfo_pts_exact_on_old_ffmpeg(monkeypatch):
    from dancecap import ingest
    m = ingest.SHOWINFO.search("[Parsed_showinfo_0 @ 0x1] n:  10 pts:  74074 pts_time:123.457 duration:")
    monkeypatch.setattr(ingest, "_PASSTHROUGH", ["-vsync", "passthrough"])
    assert ingest.showinfo_pts(m, "1/600") == round(74074 / 600, 6) == 123.456667
    assert ingest.showinfo_pts(m, None) == 123.457
    monkeypatch.setattr(ingest, "_PASSTHROUGH", ["-fps_mode", "passthrough"])
    assert ingest.showinfo_pts(m, "1/600") == 123.457  # newer ffmpeg: pts_time kept as before
