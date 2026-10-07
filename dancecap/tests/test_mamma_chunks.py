"""dancecap.mamma_chunks: chunk plan, merge (identity split of a real fit), quality gate and its pod fallback."""
import json
import shutil
import time

import numpy as np
import pytest

from dancecap import mamma_chunks as mc
from dancecap import mamma_dyn as md


def test_plan_chunks_cover_the_grid_with_equal_overlaps():
    for T, n, ov in ((1148, 6, 36), (1148, 6, 48), (176, 3, 36), (100, 1, 36)):
        ch = mc.plan_chunks(T, n, ov)
        assert ch[0][0] == 0 and ch[-1][1] == T and len(ch) == n
        assert len({b - a for a, b in ch}) == 1  # equal lengths
        for (a0, b0), (a1, b1) in zip(ch, ch[1:]):
            assert a1 < b0 and b0 - a1 >= ov  # every neighbour pair overlaps by at least `ov`


def _report(seam_p95=20.0, pelvis=10.0, swap=5.0, merged=6.0, chunk_px=(5.8, 6.0, 6.2)):
    tags = [f"c8_f{i}" for i in range(len(chunk_px))]
    role = {"seam_mm": [{"between": [a, b], "median": seam_p95 / 2, "p95": seam_p95, "pelvis_median": pelvis}
                        for a, b in zip(tags, tags[1:])],
            "betas_swap_mm": [{"chunk": t, "median": swap / 2, "p95": swap} for t in tags]}
    return {"roles": {"lead": role, "follow": json.loads(json.dumps(role))},
            "smplx": {"lead": {"median_reproj_px": merged}, "follow": {"median_reproj_px": merged}},
            "chunk_median_reproj_px": {t: {"lead": p, "follow": p} for t, p in zip(tags, chunk_px)},
            "chunk_frames": {t: 222 for t in tags}}


def test_gate_passes_normal_merge_and_catches_gross_failures():
    assert mc.gate(_report())["passed"]
    cases = {"seam p95": _report(seam_p95=90), "pelvis": _report(pelvis=55), "betas swap": _report(swap=60),
             "merged reprojection": _report(merged=9.0), "chunk(s)": _report(chunk_px=(5.8, 6.0, 15.0)),
             "no roles": {"roles": {}}}
    for word, rep in cases.items():
        g = mc.gate(rep)
        assert not g["passed"] and any(word in w for w in g["why"]), (word, g["why"])
    assert not mc.gate(_report(merged=float("nan")))["passed"]  # a NaN reprojection never passes


def _fake_take(tmp_path, monkeypatch, gate_passed: bool):
    work = tmp_path / "w"
    root = work / "mamma"
    (root / "fits" / "c8").mkdir(parents=True)
    (root / "fits" / "c8" / "fit.json").write_text("{}")
    (root / "chunks.json").write_text(json.dumps({"tag": "c8", "T": 1148, "chunks": []}))
    rep = _report() if gate_passed else _report(seam_p95=200)
    rep["gate"] = mc.gate(rep)
    (root / "chunks_merge.json").write_text(json.dumps(rep))
    monkeypatch.setattr(md, "take_work", lambda take: work)
    return root


def test_gate_command_passes_through_and_refuses_without_pod_time(tmp_path, monkeypatch):
    root = _fake_take(tmp_path, monkeypatch, gate_passed=True)
    assert mc.gate_or_fallback("x", "full") == 0
    assert json.loads((root / "chunks_gate.json").read_text())["fallback"] is None
    root = _fake_take(tmp_path / "b", monkeypatch, gate_passed=False)
    kill = tmp_path / "kill_at"
    kill.write_text(str(time.time() + 10 * 60))  # 10 min left << a full 1148-frame fit
    monkeypatch.setattr(mc, "KILL_AT", kill)
    assert mc.gate_or_fallback("x", "full") == 3
    doc = json.loads((root / "chunks_gate.json").read_text())
    assert doc["fallback"]["result"].startswith("skipped") and (root / "fits" / "c8").exists()  # merge left in place
    assert mc.gate_or_fallback("x", "none") == 3  # no fallback asked: select must not run


def test_write_job_runs_the_gate_before_select_and_budgets_the_fallback(tmp_path, monkeypatch):
    work = tmp_path / "w"
    (work / "mamma").mkdir(parents=True)
    prep = {"sets": {"c8": {"seq": "t_c8", "cams": ["01", "02", "03"]}, "h03": {"seq": "t_h03", "cams": ["01", "02"]}},
            "eval": {"seq": "t_c8", "cams": ["01", "02", "03"]},
            "grid": {"T": 1148, "t0": 30.0, "hz": 30.0, "t": list(30 + np.arange(1148) / 30.0)}}
    (work / "mamma" / "prepare.json").write_text(json.dumps(prep))
    monkeypatch.setattr(md, "take_work", lambda take: work)
    mc.plan("t", "c8", 6, 48)
    a = json.loads(mc.write_job("t", fallback="full").read_text())
    names = [c["name"] for c in a["commands"]]
    assert names.index("merge") < names.index("gate") < names.index("select")
    assert "--fallback full" in a["commands"][names.index("gate")]["cmd"]
    b = json.loads(mc.write_job("t", fallback="none").read_text())
    assert a["max_minutes"] - b["max_minutes"] >= mc.solo_minutes(1148)  # the fallback fit is in the time budget
    assert any("chunks_gate.json" in o for o in a["outputs"])
    # held-out check: one chunk refitted without cam03 from the prepared set h03; its inputs travel with the job
    d = mc.plan("t", "c8", 6, 48, holdout_cam="03", holdout_chunk=4)
    assert d["holdout"]["run_tag"] == f"h03_f{d['chunks'][4]['a']}-{d['chunks'][4]['b']}"
    j = json.loads(mc.write_job("t").read_text())
    assert any("t_h03" in i for i in j["inputs"])
    with pytest.raises(SystemExit):
        mc.plan("t", "c8", 6, 48, holdout_cam="04")  # no prepared set h04


SRC = md.PACKAGE_ROOT / "work" / "larissa_kadu" / "mamma"


@pytest.mark.skipif(not (SRC / "fits" / "c8" / "smplx" / "lead.npz").exists(),
                    reason="needs the larissa_kadu c8 fit and the SMPL-X model (local data)")
def test_merge_reproduces_a_fit_split_into_chunks_and_flags_a_shifted_chunk(tmp_path, monkeypatch):
    """A real fit cut into 3 overlapping chunks with identical values in the overlaps must merge back to itself
    (0 mm seams, 0 mm betas swap, gate passed); one chunk moved by 10 cm must fail the gate on its seams."""
    work = tmp_path / "w"
    root = work / "mamma"
    root.mkdir(parents=True)
    prep = json.loads((SRC / "prepare.json").read_text())
    seq = prep["sets"]["c8"]["seq"]
    prep["eval"] = {"seq": seq, "cams": prep["sets"]["c8"]["cams"]}
    (root / "prepare.json").write_text(json.dumps(prep))
    for d in ("ma_2d", "ma_cap"):
        shutil.copytree(SRC / d / seq, root / d / seq)
    T = prep["grid"]["T"]
    chunks = mc.plan_chunks(T, 3, 36)
    (root / "chunks.json").write_text(json.dumps({
        "tag": "c8", "T": T, "overlap": 36, "cams": prep["sets"]["c8"]["cams"], "hz": 30.0,
        "chunks": [{"a": a, "b": b, "run_tag": f"c8_f{a}-{b}"} for a, b in chunks]}))
    full = {r: dict(np.load(SRC / "fits" / "c8" / "smplx" / f"{r}.npz", allow_pickle=True)) for r in ("lead", "follow")}
    fj = (SRC / "fits" / "c8" / "fit.json").read_text()
    monkeypatch.setattr(md, "take_work", lambda take: work)

    def write(shift=None):
        for k, (a, b) in enumerate(chunks):
            d = root / "fits" / f"c8_f{a}-{b}" / "smplx"
            d.mkdir(parents=True, exist_ok=True)
            for r, f in full.items():
                g = {kk: (v[a:b] if isinstance(v, np.ndarray) and v.ndim and len(v) == T else v) for kk, v in f.items()}
                if shift is not None and k == 1:
                    g["transl"] = g["transl"] + shift
                    g["joints"] = g["joints"] + shift
                np.savez(d / f"{r}.npz", **g)
            (d.parent / "fit.json").write_text(fj)

    write()
    rep = mc.merge("x")
    for r in ("lead", "follow"):
        with np.load(root / "fits" / "c8" / "smplx" / f"{r}.npz") as m:  # closed: Windows cannot delete open files
            joints = m["joints"]
        assert np.abs(joints - full[r]["joints"][:, :55]).max() * 1000 < 0.5
        assert max(s["p95"] for s in rep["roles"][r]["seam_mm"]) < 0.5
        assert max(s["p95"] for s in rep["roles"][r]["betas_swap_mm"]) < 0.5
    assert rep["gate"]["passed"], rep["gate"]["why"]
    assert json.loads((root / "fits" / "c8" / "fit.json").read_text())["partial"] is False
    # held-out report: a fake held-out fit of chunk 1 (cam03 out) -> held-out vs in-fit residual of cam03
    a, b = chunks[1]
    doc = json.loads((root / "chunks.json").read_text())
    doc["holdout"] = {"tag": "h03", "cam": "03", "chunk": 1, "a": a, "b": b, "run_tag": f"h03_f{a}-{b}", "t": [0, 1]}
    (root / "chunks.json").write_text(json.dumps(doc))
    hd = root / "fits" / f"h03_f{a}-{b}"
    hd.mkdir(parents=True)
    (hd / "fit.json").write_text(json.dumps({"residuals": {"03": {"lead": {"median_px": 9.0}, "follow": {"median_px": 11.0}}}}))
    rep = mc.merge("x")
    h = rep["holdout"]
    assert h["status"] == "ok" and h["lead"]["held_out_median_px"] == 9.0
    assert 0 < h["lead"]["in_fit_median_px"] < 9.0 and h["follow"]["ratio"] > 1
    assert not any(p.name.startswith("_holdout_slice") for p in (root / "fits").iterdir())
    write(shift=np.array([0.10, 0.0, 0.0]))
    rep = mc.merge("x")
    assert not rep["gate"]["passed"] and any("seam" in w for w in rep["gate"]["why"])
