"""dancecap.farm.analysis: the pod jobs of FARM.md 0.10 (stitch, analyse, stitch+analyse, prepare) in the worker's own conventions
(kind `analyse` on CPU pods, `after`, archive inputs of chunks/<id>/out/result.tar), the optional queue hold, the evidence step.
No pods, no Runpod calls."""
import json

import pytest

from _farm_fixtures import ANCHOR, make_spec, patch_root, write_motion, write_prepare
from _needs import take_file
from dancecap.farm import analysis as A
from dancecap.farm import chunks as C
from dancecap.farm import cli
from dancecap.farm import timeline as tl
from dancecap.farm import worker
from dancecap.farm.state import State


def chunk(cid, k0, k1, core0, core1, take=None, status="planned", **kw):
    d = {"id": cid, "k0": k0, "k1": k1, "core_k0": core0, "core_k1": core1, "take": take or f"dance_{cid}", "status": status,
         "origin": "planned", "cams": ["01", "02"], "stages_done": [], "kind": "full"}
    d.update(kw)
    return d


def spec_with_cfg(**kw):
    s = make_spec(**kw)
    s.cfg.update({"stitch": {"blend_s": 2.0, "edge_margin_s": 0.2, "gate_mm": {"body_p95": 60}, "mono": [[1.0, 2.0]]},
                  "reuse": {"take": "ref", "motion_raw": "smplx", "motion_refined": "rf"}})
    return s


# ------------------------------------------------------------------------------------------------------ registry


def test_default_analyses_are_registered_on_the_workers_command_table():
    assert set(A.DEFAULT_ANALYSES) <= set(A.ANALYSES)
    assert all(a.base in worker.ANALYSIS for a in A.ANALYSES.values())        # one place for the base commands
    assert A.ANALYSES["pose_quality"].scope == "chunk" and A.ANALYSES["timing"].optional is False
    with pytest.raises(ValueError, match="unknown analysis"):
        A.resolve_analyses(["nope"])
    A.register_analysis(A.Analysis("asr_test", "timing", "--x", 5, gpu=True, outputs=("asr/",)))
    try:
        assert A.resolve_analyses(["asr_test"])[0].gpu
    finally:
        A.ANALYSES.pop("asr_test")


def test_kinds_split_into_cpu_and_gpu_pods_and_cpu_jobs_use_the_workers_analyse_kind():
    assert set(A.CPU_KINDS) | set(A.GPU_KINDS) == set(C.KINDS) and not set(A.CPU_KINDS) & set(A.GPU_KINDS)
    assert {"prepare", "stitch", "analyse", "stitch_analyse"} <= set(C.KINDS)
    assert A.WORKER_KIND == "analyse" and not worker.needs_gpu({"kind": A.WORKER_KIND})   # pods.launch_cpu(kinds="analyse")


# ------------------------------------------------------------------------------------------------- the final job


def test_final_job_is_one_cpu_pod_job_over_the_chunk_list(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg(reference="ref", ref_k=(0, 89))
    cl = [chunk("g1", 90, 209, 100, 199), chunk("g2", 190, 299, 200, 289), chunk("bad", 0, 9, 0, 9, status="failed")]
    job = A.build_final_job(spec, cl, None, hash_inputs=False, code={"git_sha": "x"})
    json.dumps(job)                                                           # serialisable
    assert job["kind"] == "analyse" and job["what"] == "stitch_analyse" and job["needs_gpu"] is False and job["id"] == "final"
    assert not worker.needs_gpu(job)
    assert job["after"] == ["g1", "g2"] and job["chunks"] == ["g1", "g2"]    # failed chunks are not stitched
    ins = A.build_final_job(spec, cl, None, hash_inputs=True, code={}, analyses=("timing",))["inputs"]
    assert {"path": "chunks/g1/out/result.tar", "archive": True} in ins and {"path": "chunks/g2/out/result.tar", "archive": True} in ins
    names = [s["name"] for s in job["steps"]]
    assert names[:2] == ["stitch_prep", "stitch"] and names[-1] == "evidence"
    assert [n for n in names if n in A.DEFAULT_ANALYSES] == list(A.DEFAULT_ANALYSES)
    assert "pose_quality" not in names                                        # chunk-scope analyses run per chunk
    stitch = job["steps"][1]["cmd"]
    assert stitch.startswith("python -m dancecap.stitch dance --chunks work/dance/farm/stitch_list.json --partial")
    assert "[ $rc -eq 2 ]" in stitch and "exit 0" in stitch                  # a failed seam gate is a coverage fact, not a job failure
    assert "--beats work/dance/song/beats_studio.json" in next(s["cmd"] for s in job["steps"] if s["name"] == "timing")
    assert "--smplx-dir rf" in next(s["cmd"] for s in job["steps"] if s["name"] == "counterbalance")
    assert next(s for s in job["steps"] if s["name"] == "timing")["optional"] is False
    assert next(s for s in job["steps"] if s["name"] == "floorcraft")["optional"] is True
    assert next(s for s in job["steps"] if s["name"] == "evidence")["cmd"].endswith("chunk-evidence final --stage final")
    outs = {o["path"] for o in job["outputs"]}
    assert {"dancecap/work/dance/smplx/", "dancecap/work/dance/stitch.json", "dancecap/work/dance/farm/evidence.npz"} <= outs
    assert all(o["path"].startswith("dancecap/work/dance/") for o in job["outputs"])
    assert job["env"]["DANCECAP_FARM_DANCE"] == "dance" and job["cwd"] == "dancecap"
    assert len(job["stitch_chunks"]) == 2


def test_chunks_with_local_motion_travel_as_inputs_the_rest_are_after(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg(reference="ref", ref_k=(0, 89))
    w = tmp_path / "work"
    write_motion(w / "dance_s1" / "smplx", 90, 199)
    write_motion(w / "dance_s1" / "rf", 90, 199)                             # S1 segment: fit + refine already on this machine
    write_motion(w / "dance_g2" / "smplx", 190, 299)                          # raw only: a refine job is still needed
    cl = [chunk("s1", 90, 199, 100, 195, origin="s1"), chunk("g2", 190, 299, 196, 299)]
    job = A.build_final_job(spec, cl, None, hash_inputs=False, code={})
    assert job["local_chunks"] == ["s1"] and job["after"] == ["g2"]
    assert A.needs_met(job, {"g2": "running"}) is False
    assert A.needs_met(job, {"g2": "done"}) and A.needs_met(job, {"g2": "pulled"}) and A.needs_met(job, {"g2": "stitched"})
    ready, held = A.ready_jobs([job, {"id": "x", "after": []}], {"g2": "queued"})
    assert [j["id"] for j in ready] == ["x"] and [j["id"] for j in held] == ["final"]


def test_with_stitch_false_and_single_take_analysis_jobs(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg()
    job = A.build_final_job(spec, [chunk("g1", 0, 99, 0, 99)], None, with_stitch=False, hash_inputs=False, code={})
    assert job["what"] == "analyse" and [s["name"] for s in job["steps"]][0] == "timing"
    one = A.build_analysis_job(spec, "dance_g1", ["pose_quality", "timing"], None, hash_inputs=False, code={}, after=["g1"])
    assert one["id"] == "an-dance_g1" and one["kind"] == "analyse" and one["take"] == "dance_g1" and one["after"] == ["g1"]
    assert [s["name"] for s in one["steps"]] == ["pose_quality", "timing", "evidence"]
    assert one["steps"][0]["cmd"] == "python -m dancecap.pose_quality dance_g1 --smplx-dir rf --tag rf2_noslip"
    assert {"path": "chunks/g1/out/result.tar", "archive": True} in one["inputs"]
    assert {"path": "dancecap/work/dance_g1/farm/evidence.npz"} in one["outputs"]
    gpu = A.register_analysis(A.Analysis("gpu_one", "timing", "", 5, gpu=True))
    try:
        j = A.build_analysis_job(spec, "dance", ["gpu_one"], None, hash_inputs=False, code={})
        assert j["needs_gpu"] is True and worker.needs_gpu(j)                 # the worker leaves it to GPU pods
    finally:
        A.ANALYSES.pop(gpu.name)


def test_prepare_and_chunk_analyse_kinds_through_build_job(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg()
    ch = C.Chunk.from_dict(chunk("g1", 0, 99, 0, 99, estimate={"front_min": 5, "fit_min": 5, "refine_min": 1}))
    job = C.build_job(ch, spec, None, kind="prepare", hash_inputs=False, code={})
    assert job["kind"] == "analyse" and job["what"] == "prepare" and job["needs_gpu"] is False and job["id"] == "prep-g1"
    assert [s["name"] for s in job["steps"]] == ["chunk_prepare", "evidence"]
    assert {"path": "chunks/g1/out/result.tar", "archive": True} in job["inputs"]
    an = C.build_job(ch, spec, None, kind="analyse", hash_inputs=False, code={})
    assert an["what"] == "analyse" and an["after"] == ["g1"] and [s["name"] for s in an["steps"]] == ["pose_quality", "evidence"]
    with pytest.raises(ValueError, match="list of chunks"):
        C.build_job(ch, spec, None, kind="stitch", hash_inputs=False, code={})


def test_chunk_jobs_end_with_the_evidence_step_and_declare_it(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg()
    ch = C.Chunk.from_dict(chunk("g1", 0, 99, 0, 99, stages_done=["front"], estimate={"front_min": 5, "fit_min": 5, "refine_min": 1,
                                                                                         "total_min": 20}))
    job = C.build_job(ch, spec, None, hash_inputs=False, code={})
    assert job["stages"] == ["fit", "refined"] and job["steps"][-1]["name"] == "evidence"
    assert job["steps"][-1]["cmd"].endswith("chunk-evidence g1 --stage refined")
    assert {"path": "dancecap/work/dance_g1/farm/evidence.npz"} in job["outputs"]
    assert any(s["name"] == "quality" and s["optional"] for s in job["steps"])   # per-chunk pose_quality on the pod that holds the inputs


# ------------------------------------------------------------------------------------------------ the stitch list


def test_stitch_list_orders_chunks_and_fills_ma2d_when_the_prepare_exists(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg(reference="ref", ref_k=(100, 199))
    w = tmp_path / "work"
    cl = [chunk("g2", 190, 299, 200, 299), chunk("g0", 0, 109, 0, 99)]
    doc = A.stitch_list(spec, cl, work_root=w)
    assert [c["name"] for c in doc["chunks"]] == ["g0", "ref", "g2"]
    ref = next(c for c in doc["chunks"] if c.get("reference"))
    assert ref["k"] == [100, 199] and ref["src"] == "ref/smplx" and ref["refined"] == "ref/rf" and ref["margin_s"] == 0.5
    g2 = next(c for c in doc["chunks"] if c["name"] == "g2")
    assert g2["src"] == "dance_g2/smplx" and "ma_2d" not in g2                  # no prepare.json of dance_g2 on this machine
    write_prepare(w / "dance_g2", 190, 299, ["01", "02"])
    g2 = next(c for c in A.stitch_list(spec, cl, work_root=w)["chunks"] if c["name"] == "g2")
    assert g2["ma_2d"] == "dance_g2/mamma/ma_2d/dance_g2_all" and g2["prepare"] == "dance_g2/mamma/prepare.json"
    assert doc["anchor"] == ANCHOR and doc["hz"] == 30.0 and doc["gate_mm"] == {"body_p95": 60} and doc["mono"] == [[1.0, 2.0]]


# -------------------------------------------------------------------------------------- queue with a fake uploader


def _queued_world(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    st = State("dance")
    st.put_chunks([chunk("g1", 0, 99, 0, 99), chunk("g2", 100, 199, 100, 199)])   # one call: put_chunks drops absent planned chunks
    jd = tmp_path / "work" / "dance" / "farm" / "jobs"
    jd.mkdir(parents=True, exist_ok=True)
    for cid in ("g1", "g2"):
        (jd / f"{cid}.json").write_text(json.dumps({"id": cid, "kind": "full", "after": []}))
        st.set_job(cid, chunk=cid, kind="full", status="planned")
    (jd / "final.json").write_text(json.dumps({"id": "final", "kind": "analyse", "what": "stitch_analyse", "after": ["g1", "g2"]}))
    st.set_job("final", kind="stitch_analyse", status="planned", after=["g1", "g2"])
    got = []
    monkeypatch.setattr(cli, "HOOKS", {"queue_upload": [lambda dance, jobs, args: got.append([j["id"] for j in jobs]) or
                                                          {j["id"]: {"ok": True} for j in jobs}]})
    return st, got


def test_queue_uploads_everything_the_worker_enforces_after(tmp_path, monkeypatch):
    st, got = _queued_world(tmp_path, monkeypatch)
    r = cli.queue_jobs("dance")
    assert sorted(r["queued"]) == ["final", "g1", "g2"] and [sorted(x) for x in got] == [["final", "g1", "g2"]] and r["held"] == []
    s = st.load()
    assert s["chunks"]["g1"]["status"] == "queued" and s["jobs"]["final"]["status"] == "queued"
    assert cli.queue_jobs("dance")["queued"] == []                              # nothing is queued twice


def test_queue_hold_keeps_the_final_job_until_its_chunks_are_done(tmp_path, monkeypatch):
    st, got = _queued_world(tmp_path, monkeypatch)
    r = cli.queue_jobs("dance", hold=True)
    assert sorted(r["queued"]) == ["g1", "g2"] and r["held"] == [{"id": "final", "after": ["g1", "g2"]}]
    assert st.load()["jobs"]["final"]["status"] == "planned"
    st.set_chunk("g1", status="done")
    r = cli.release_ready("dance")                                              # the watcher's per-cycle call
    assert r["queued"] == [] and r["held"] == [{"id": "final", "after": ["g2"]}]
    st.set_chunk("g2", status="pulled")
    r = cli.release_ready("dance")
    assert r["queued"] == ["final"] and got[-1] == ["final"] and r["held"] == []
    assert cli.queue_jobs("dance", hold=True)["queued"] == []


def test_queue_local_only_uploads_nothing(tmp_path, monkeypatch):
    st, got = _queued_world(tmp_path, monkeypatch)
    r = cli.queue_jobs("dance", local_only=True)
    assert got == [] and sorted(r["ready"]) == ["final", "g1", "g2"] and r["queued"] == []


# ------------------------------------------------------------------------------------------ the pod-side steps


def _manifest(vol, cid, status="ok", world_id=None, missing=()):
    out = vol / "chunks" / cid / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps({"id": cid, "status": status, "world_id": world_id, "missing": list(missing)}))


def test_check_upstream_refuses_unfinished_or_partial_chunks(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    vol = tmp_path / "vol"
    with pytest.raises(FileNotFoundError, match="not finished"):
        A.check_upstream(["g1"], vol, None, "dance")
    _manifest(vol, "g1", status="partial", missing=["dancecap/work/x/smplx/"])
    with pytest.raises(RuntimeError, match="missing"):
        A.check_upstream(["g1"], vol, None, "dance")
    _manifest(vol, "g1", world_id="w1")
    _manifest(vol, "g2", world_id="w1")
    assert A.check_upstream(["g1", "g2"], vol, None, "dance")["chunks"] == 2


def test_stitch_prep_writes_the_stitch_list_and_the_world_frame(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg(reference="ref", ref_k=(0, 89))
    monkeypatch.setattr(tl.DanceSpec, "load", classmethod(lambda cls, name=None: spec))
    vol, ws = tmp_path / "vol", tmp_path / "ws"
    _manifest(vol, "g1")
    for take in ("dance_g1", "ref"):                                                # what the worker's archive inputs extracted
        (ws / "dancecap" / "work" / take / "smplx").mkdir(parents=True)
        (ws / "dancecap" / "work" / take / "smplx" / "lead.npz").write_bytes(b"x")
    world = tmp_path / "world"
    (world / "ref" / "cameras").mkdir(parents=True)
    (world / "ref" / "cameras" / "world_from_sfm.json").write_text('{"s": 1}')
    monkeypatch.setenv("HM_WORLD_DIR", str(world))
    job = {"id": "final", "dance": "dance", "what": "stitch_analyse", "after": ["g1"], "world_id": None,
           "stitch_chunks": [chunk("g1", 80, 199, 90, 189)]}
    jp = tmp_path / "final.json"
    jp.write_text(json.dumps(job))
    r = A.stitch_prep("final", "dance", job_path=jp, volume=vol, ws=ws)
    assert r["world_frame"] is True and r["stitch_list"] == 2                      # the reference + g1
    doc = json.loads((tmp_path / "work" / "dance" / "farm" / "stitch_list.json").read_text())
    assert {c["name"] for c in doc["chunks"]} == {"ref", "g1"}
    assert (tmp_path / "work" / "dance" / "cameras" / "world_from_sfm.json").read_text() == '{"s": 1}'
    (ws / "dancecap" / "work" / "dance_g1" / "smplx" / "lead.npz").unlink()          # a chunk that was not extracted is an error
    with pytest.raises(FileNotFoundError, match="not extracted"):
        A.stitch_prep("final", "dance", job_path=jp, volume=vol, ws=ws)


def test_chunk_evidence_step_for_a_chunk_and_for_the_stitched_dance(tmp_path, monkeypatch):
    patch_root(monkeypatch, tmp_path)
    spec = spec_with_cfg(reference="ref", ref_k=(0, 89))
    monkeypatch.setattr(tl.DanceSpec, "load", classmethod(lambda cls, name=None: spec))
    w = tmp_path / "work"
    write_prepare(w / "dance_g1", 90, 179, ["01", "02", "03"])
    write_motion(w / "dance_g1" / "smplx", 90, 179, flag=3, n_cams=3)
    write_motion(w / "dance_g1" / "rf", 90, 179, flag=3, n_cams=3)
    jp = tmp_path / "g1.json"
    jp.write_text(json.dumps({"id": "g1", "take": "dance_g1", "chunk": {"id": "g1"}}))
    r = A.chunk_evidence("g1", "refined", "dance", job_path=jp)
    assert r["stitched"] is False and r["frames"] == 90
    assert tl.Frames.from_npz(w / "dance_g1" / "farm" / "evidence.npz", spec.grid).level.max() == tl.L_REFINED
    # the dance's own job: the stitched motion + the reference span -> evidence.npz (final) + coverage_pod.json
    write_motion(w / "ref" / "smplx", 0, 89, flag=3, n_cams=8)
    write_motion(w / "ref" / "rf", 0, 89, flag=3, n_cams=8)
    for d in ("smplx", "rf"):
        write_motion(w / "dance" / d, 90, 179, flag=3, n_cams=3)
    jf = tmp_path / "final.json"
    jf.write_text(json.dumps({"id": "final", "take": "dance", "chunk": {"id": "final"}}))
    r = A.chunk_evidence("final", "final", "dance", job_path=jf)
    assert r["stitched"] is True and r["coverage"]["counts"]["complete"] == 6
    pod = json.loads((w / "dance" / "farm" / "coverage_pod.json").read_text())
    assert pod["counts"]["complete"] == 6 and pod["dance"] == "dance"
    fin = tl.Frames.from_npz(w / "dance" / "farm" / "evidence.npz", spec.grid)
    assert (fin.final_level[:180] >= tl.L_REFINED).all() and (fin.final_level[180:] == 0).all()


def test_cli_analyse_lists_and_plans_without_running_anything(capsys):
    assert cli.main(["analyse", "--list"]) == 0
    out = capsys.readouterr().out
    assert "timing" in out and "counterbalance" in out and "CPU" in out
    if not take_file("larissa_kadu_full").is_file():
        pytest.skip("take config not in this checkout: larissa_kadu_full (the planning half needs it)")
    assert cli.main(["--dance", "larissa_kadu_full", "analyse", "--analyses", "timing", "--dry-run"]) == 0
    assert "CPU pod" in capsys.readouterr().out
