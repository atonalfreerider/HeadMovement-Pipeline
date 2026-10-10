"""farm.recap_analysis (the beat-free analysis jobs of the class recap) and stitch_provisional (the PROVISIONAL.md writer)."""
import json
from types import SimpleNamespace

from dancecap import stitch_provisional as sp
from dancecap.farm import analysis as A
from dancecap.farm import recap_analysis as RA


def _spec():
    return SimpleNamespace(name="larissa_kadu_recap", refined_dir="smplx_rf2_noslip", grid=SimpleNamespace(k0=0, k1=10814))


def test_jobs_follow_the_stitch_job_and_use_no_beats(monkeypatch):
    monkeypatch.setattr("dancecap.farm.chunks.code_info", lambda: {"git_sha": "x"})
    jobs = RA.build_jobs(_spec())
    assert [j["id"] for j in jobs] == ["an-timing-w", "an-physics-m", "an-floor"]
    for j in jobs:
        assert j["kind"] in ("analyse", "analyse_mem") and j["needs_gpu"] is False and j["after"] == ["final"]
        assert j["inputs"] == [{"path": "chunks/final/out/result.tar", "archive": True}]
        assert j["world_id"] is None
        cmds = " ".join(s["cmd"] for s in j["steps"])
        assert "--beats" not in cmds and "beats_studio" not in cmds
        assert all(o["path"].startswith("dancecap/work/larissa_kadu_recap/") for o in j["outputs"])
    t = {j["id"]: j for j in jobs}
    assert "python -m dancecap.timing_free larissa_kadu_recap --mc 60" in t["an-timing-w"]["steps"][0]["cmd"]
    assert t["an-physics-m"]["steps"][0]["cmd"].startswith("python -m dancecap physics larissa_kadu_recap")
    assert t["an-physics-m"]["kind"] == "analyse_mem" and t["an-timing-w"]["kind"] == "analyse"      # only the 64 GB pod takes physics
    assert [s["name"] for s in t["an-floor"]["steps"]] == ["counterbalance", "floorcraft"]
    assert all("--smplx-dir smplx_rf2_noslip" in s["cmd"] for s in t["an-floor"]["steps"])
    assert [o["path"].rsplit("/", 2)[-2] for o in t["an-floor"]["outputs"]] == ["counterbalance", "floorcraft"]


def test_only_filters_and_registry():
    assert [j["id"] for j in RA.build_jobs(_spec(), only={"an-physics-m"})] == ["an-physics-m"]
    assert "timing_free" in A.ANALYSES and "physics_free" in A.ANALYSES
    assert "{beats}" not in A.ANALYSES["timing_free"].args + A.ANALYSES["physics_free"].args


def _stitch_doc(passed=False):
    return {"grid": {"T": 300, "hz": 30.0, "t": [0.0, 9.97]}, "thresholds": {"body_p95": 60.0},
            "gate": {"passed": passed}, "outputs": {"forced": True, "sha256": {}},
            "coverage": {"frames": {"complete": 250, "partial": 50, "none": 0}},
            "flags": {"lead": {"counts": {"absent": 0, "mono": 0, "two_view": 10, "multi": 290}},
                      "follow": {"counts": {"absent": 0, "mono": 0, "two_view": 10, "multi": 290}}},
            "seams": [{"name": "a|b", "window_t": [4.0, 6.0], "gate": {"passed": False, "why": ["lead a|b: r_wrist p95 99 mm > 75"]},
                       "refit": {"window_t": [0.0, 12.0], "window_k": [0, 360]}},
                      {"name": "b|c", "window_t": [7.0, 9.0], "gate": {"passed": True, "why": []}}]}


def test_provisional_report_lists_every_failed_gate_and_weak_stretch():
    st = _stitch_doc()
    rows = sp.seam_rows(st)
    md = sp.render("t", st, rows, {"smplx/lead.npz": {"sha256": "ab" * 32, "same_as_pod_record": True}}, "because",
                   [{"chunk": "g0000", "t_s": [10, 30], "what": "follow low confidence"}], "2026-10-09T00:00:00")
    assert "FAILED on 1 of 2 seams" in md
    assert "lead a|b: r_wrist p95 99 mm > 75" in md
    assert "| b|c | 7.00 to 9.00 | pass | - |" in md
    assert "g0000 [10, 30] s: follow low confidence" in md
    assert "refit 0.00 to 12.00 s" in md
    assert "smplx/lead.npz: " + "ab" * 32 in md
    json.dumps(rows)


def test_check_arrays_compares_with_the_pod_record(tmp_path):
    (tmp_path / "smplx").mkdir()
    (tmp_path / "smplx" / "lead.npz").write_bytes(b"abc")
    good = sp.sha256(tmp_path / "smplx" / "lead.npz")
    st = {"outputs": {"sha256": {"smplx/lead.npz": good, "smplx/follow.npz": "00"}}}
    r = sp.check_arrays(tmp_path, st)
    assert r["smplx/lead.npz"]["same_as_pod_record"] is True
    assert r["smplx/follow.npz"]["same_as_pod_record"] is False and r["smplx/follow.npz"]["sha256"] is None


def test_mem_pod_is_a_memory_flavor_that_only_takes_the_mem_kind(monkeypatch):
    seen = {}

    def fake_launch_cpu(n, **kw):
        seen.update(kw, n=n)
        return []                                   # nothing placed: nothing to record
    monkeypatch.setattr("dancecap.farm.pods.launch_cpu", fake_launch_cpu)
    RA.launch_mem_pod("larissa_kadu_recap")
    assert seen["flavors"][0] in ("cpu5m", "cpu3m") and seen["kinds"] == RA.MEM_KIND == "analyse_mem" and seen["n"] == 1
    from dancecap.farm import pods
    assert pods.CPU_FLAVORS[seen["flavors"][0]][1] >= 8          # GB per vCPU


def test_plain_capture_validation_separates_expected_from_real_failures():
    from dancecap import recap_export as RE
    checks = [{"name": "capture.json", "ok": True, "detail": None},
              {"name": "lead albedo", "ok": False, "detail": "no smplx_albedo entry"},
              {"name": "audio", "ok": False, "detail": "no audio (silent clock)"},
              {"name": "floorcraft.json", "ok": False, "detail": "missing"},
              {"name": "lead motion frames", "ok": False, "detail": "10 vs 11"}]
    c = RE.classify(checks)
    assert [x["name"] for x in c["expected"]] == ["lead albedo", "audio"]
    assert [x["name"] for x in c["failed"]] == ["floorcraft.json", "lead motion frames"]
    assert c["passed"] == ["capture.json"]
