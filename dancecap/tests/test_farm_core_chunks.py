"""dancecap.farm.chunks: gap planner, S1 registration, job specs, chunk take TOMLs."""
import json
import tomllib

import numpy as np
import pytest

import dancecap.farm.chunks as C
from _farm_core_fixtures import ANCHOR, make_spec, patch_root, write_motion, write_prepare
from dancecap.farm import timeline as tl
from dancecap.farm import world as W

FULL = tl.Grid(30.151176844227862, -998, 4645)


def full_spec():
    return tl.DanceSpec(name="d", grid=FULL, reference="ref", ref_k=(0, 1147), cams=["01", "02", "03", "04", "05", "06", "07", "09"],
                        cfg={"source_dir": "../../dances/Cap", "alignment": "../../prototype/a.json",
                             "dance": {"level": "pro"}, "dancers": {"lead": {"name": "L"}, "follow": {"name": "F"}}})


def synth_coverage(spec, refined, partial=()):
    """Coverage with the given (k0, k1) spans refined + stitched (complete), everything else nothing."""
    fr = tl.Frames(spec.grid)
    fr.max_cams[:] = 8
    for a, b in refined:
        fr.apply(np.arange(a, b + 1), tl.L_REFINED, n_cams=8, flag=tl.FLAG_MULTI, final=True)
    return tl.Coverage(spec.name, spec.grid, tl.fold_cells(fr), frames=fr)


SPANS8 = {"01": (-50.0, 500.0), "02": (-50.0, 500.0), "03": (-50.0, 500.0), "04": (-50.0, 500.0), "05": (-50.0, 500.0),
          "06": (-50.0, 500.0), "07": (-50.0, 500.0), "09": (-50.0, 500.0)}


# ------------------------------------------------------------------------------------------------ splitting


def test_split_gap_examples_of_the_task():
    assert C.split_gap(998) == [998]                                   # 33.3 s -> one chunk
    four = C.split_gap(3498)                                           # 116.6 s -> four of 29.2 s
    assert len(four) == 4 and sum(four) == 3498 and max(four) - min(four) <= 1
    assert C.split_gap(15) == [15]                                     # 0.5 s
    assert C.split_gap(1080) == [1080]                                 # exactly 36 s
    assert C.split_gap(1081) == [1081]                                 # 36.03 s: no 24-36 s split exists, 1 x 36.03 beats 2 x 18
    assert C.split_gap(1200) == [600, 600]                             # 40 s: a tie between 1 x 40 and 2 x 20 goes to more chunks


def test_split_gap_properties():
    for n in list(range(1, 4000, 7)) + [1080, 1081, 1200, 1800, 2160, 2161, 3498, 4496]:
        parts = C.split_gap(n)
        assert sum(parts) == n and all(p > 0 for p in parts)
        assert max(parts) - min(parts) <= 1                            # equal chunks
        L = n / 30
        if L > 36:
            lo, hi = np.ceil(L / 36 - 1e-9), np.floor(L / 24 + 1e-9)
            if lo <= hi:                                                # an exact 24-36 s split exists: it is used
                assert all(24 * 30 - 1 <= p <= 36 * 30 + 1 for p in parts), (n, parts)
            assert all(p / 30 <= 40 for p in parts)


# ------------------------------------------------------------------------------------------------ planning


def test_the_missing_150_seconds_of_the_full_dance():
    spec = full_spec()
    cov = synth_coverage(spec, [(0, 1147)])
    assert cov.gaps() == [(-998, -1), (1148, 4645)]
    chunks, rep = C.plan_gaps(cov, spec, spans=SPANS8)
    assert [c.id for c in chunks] == ["g0000", "g2146", "g3020", "g3895", "g4770"]
    assert [c.core_s if hasattr(c, "core_s") else round(c.core_T / 30, 1) for c in chunks] == [33.3, 29.1, 29.2, 29.2, 29.1]
    cores = [(c.core_k0, c.core_k1) for c in chunks]
    assert cores[0] == (-998, -1)
    assert [b + 1 for a, b in cores[1:-1]] == [a for a, b in cores[2:]]          # contiguous, no hole, no double cover
    assert cores[1][0] == 1148 and cores[-1][1] == 4645
    assert rep["gap_s"] == [33.27, 116.6] and rep["chunks"] == 5
    first, second = chunks[0], chunks[1]
    assert (first.k0, first.k1) == (-998, 119)                                    # no overlap before the dance, 4 s into the 38 s span
    assert (second.k0, second.k1) == (1148 - 120, second.core_k1 + 120)           # 4 s into the done span and into the next chunk
    assert (chunks[-1].k1, chunks[-1].ov_right) == (4645, 0)
    assert all(c.world_id is None and c.kind == "full" and c.origin == "planned" for c in chunks)
    assert chunks[0].take == "d_g0000" and chunks[0].estimate["total_min"] > 0


def test_mid_gaps_tiny_gaps_and_claims():
    spec = full_spec()
    cov = synth_coverage(spec, [(-998, 1999), (2015, 4645)])                     # a 15-frame (0.5 s) hole in the middle
    chunks, rep = C.plan_gaps(cov, spec, spans=SPANS8)
    assert [(c.core_k0, c.core_k1) for c in chunks] == [(2000, 2014)]
    c = chunks[0]
    assert c.T == 12 * 30 and c.k0 < 2000 - 100 and c.k1 > 2014 + 100             # grown to the 12 s minimum window
    assert c.k0 >= -998 and c.k1 <= 4645
    # claimed frames are left alone, the rest of the gap is planned
    cov = synth_coverage(spec, [(0, 1147)])
    chunks, rep = C.plan_gaps(cov, spec, spans=SPANS8, claimed=[(1148, 2021, "x")])
    assert [(c.core_k0) for c in chunks][:2] == [-998, 2022] and len(chunks) == 4   # g0000 + 3 chunks for the remaining 87.5 s
    assert rep["claimed_skipped"] == [(1148, 2021, "x")]
    # two separate gaps are planned separately
    cov = synth_coverage(spec, [(-998, 99), (1000, 2999)])
    chunks, _ = C.plan_gaps(cov, spec, spans=SPANS8)
    assert len(chunks) == 3 and (chunks[0].core_k0, chunks[0].core_k1) == (100, 999)           # 30 s, then 54.9 s -> 2 x 27.4
    assert chunks[1].core_k0 == 3000 and chunks[-1].core_k1 == 4645
    assert (chunks[0].ov_left, chunks[0].ov_right) == (120, 120)


def test_cameras_of_a_chunk_follow_the_recording_spans():
    spec = full_spec()
    spans = {"01": (-10.0, 500.0), "04": (0.0, 40.0), "09": (35.0, 36.0), "07": (50.0, 500.0)}
    g = spec.grid
    k0, k1 = g.k_of(30.0), g.k_of(45.0)
    assert C.chunk_cams(spec, k0, k1, spans) == ["01", "04"]                      # 09 records 1 s only, 07 starts after
    assert C.chunk_cams(spec, g.k_of(60.0), g.k_of(70.0), spans) == ["01", "07"]


def test_unfixable_spans_are_not_planned():
    spec = make_spec(reference=None)
    fr = tl.Frames(spec.grid)
    ks = np.arange(spec.grid.k0, spec.grid.k0 + 60)
    fr.apply(ks, tl.L_REFINED, n_cams=1, flag=tl.FLAG_MONO, final=True)
    fr.max_cams[:60] = 1
    cov = tl.Coverage("dance", spec.grid, tl.fold_cells(fr), frames=fr)
    for mode in ("missing", "all"):
        chunks, _ = C.plan_gaps(cov, spec, spans={"01": (0.0, 99.0)}, mode=mode)
        assert chunks and all(c.core_k0 >= spec.grid.k0 + 60 for c in chunks), mode   # the mono span is flagged, never re-planned
    assert cov.cells[0]["state"] == "partial" and "mono" in cov.cells[0]["reasons"]


# ------------------------------------------------------------------------------------------------ S1


def test_s1_segments_become_chunks_with_cores_cut_at_the_overlaps(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    spec = full_spec()
    for sid, (k0, T, cams) in {"a": (-998, 1119, ["01", "02"]), "b": (1027, 1286, ["01", "02", "03"]),
                               "c": (2194, 1284, ["01", "02"]), "d": (3360, 1286, ["01", "02"])}.items():
        write_prepare(root / "work" / f"d_{sid}", k0, T, cams, anchor=spec.grid.anchor)
    spec.segments = {s: {} for s in "abcd"}
    write_motion(root / "work" / "d_b" / "smplx", range(1027, 1100), anchor=spec.grid.anchor)
    monkeypatch.setattr(C, "camera_spans", lambda s: SPANS8)
    cs = {c.id: c for c in C.register_s1(spec, "w1")}
    assert (cs["a"].core_k0, cs["a"].core_k1) == (-998, -1)                      # the reference span (k 0..1147) is nobody's
    assert (cs["b"].core_k0, cs["b"].core_k1) == (1148, 2252) and (cs["c"].core_k0, cs["c"].core_k1) == (2253, 3418)
    assert (cs["d"].core_k0, cs["d"].core_k1) == (3419, 4645)
    assert (cs["b"].k0, cs["b"].k1) == (1027, 2312) and cs["b"].cams == ["01", "02", "03"] and cs["b"].world_id == "w1"
    assert cs["a"].stages_done == ["front"] and cs["b"].stages_done == ["front", "fit"] and cs["a"].origin == "s1"
    assert cs["a"].take == "d_a" and cs["a"].estimate["total_min"] > 0
    cov_cover = sorted((c.core_k0, c.core_k1) for c in cs.values())
    assert [b + 1 for a, b in cov_cover[:-1]][1:] == [a for a, b in cov_cover[1:]][1:]


# ------------------------------------------------------------------------------------------------ job specs


class FakeWorld:
    id, sha256, pose_id = "w" * 12, "w" * 64, "p" * 12

    def load_json(self, rel):
        return {"mode": "refine", "filter": {"root_hz": 5.0, "rot_hz": 8.0},
                "refine": {"iters": 400, "w_floor": 1.0, "floor_mm": 5.0, "w_feet": 1.0, "w_slip": 0.0, "slip_max_v": 0.15,
                           "root_xz": False, "w_anchor": 1.0, "anchor_mm": 10.0}}


def test_job_spec_of_a_planned_chunk_has_every_stage_and_resolved_commands():
    spec = full_spec()
    ch = C.Chunk(id="g2146", k0=1028, k1=2141, core_k0=1148, core_k1=2021, cams=["01", "02"], take="d_g2146", world_id="w")
    ch.estimate = C.estimate(ch, spec, SPANS8)
    job = C.build_job(ch, spec, FakeWorld(), hash_inputs=False, code={"git_sha": "abc", "dirty": False, "bundle_sha256": None})
    assert job["schema"] == "dancecap.farm.job/1" and job["id"] == "g2146" and job["stages"] == ["front", "fit", "refined"]
    assert job["world_id"] == "w" * 12 and job["world_sha256"] == "w" * 64 and job["pose_id"] == "p" * 12
    names = [s["name"] for s in job["steps"]]
    assert names[:5] == ["chunk_take", "chunk_prep", "front", "fit", "refine"]
    for s in job["steps"]:
        assert "{chunk}" not in s["cmd"] and "{take}" not in s["cmd"] and "{refined_dir}" not in s["cmd"], s
        assert isinstance(s["timeout_min"], float) and s["timeout_min"] > 0
    cmd = {s["name"]: s for s in job["steps"]}
    assert "front_full run --segs g2146" in cmd["front"]["cmd"] and "chunk-fit g2146" in cmd["fit"]["cmd"]
    assert "--src smplx --dst smplx_rf2_noslip" in cmd["refine"]["cmd"] and "--root-hz 5.0 --rot-hz 8.0" in cmd["refine"]["cmd"]
    assert cmd["fit"]["env"]["MAMMA_FREEZE_BETAS"] == "1"
    assert cmd["fit"]["env"]["MAMMA_INIT_BETAS"] == "{WS}/dancecap/work/worlds/d/" + "w" * 12 + "/shared/betas.json"
    assert any(p.endswith("/smplx/") for p in cmd["fit"]["save"]) and "dancecap/work/d_g2146/" in cmd["fit"]["save"][0]
    outs = [o["path"] for o in job["outputs"]]
    assert "dancecap/work/d_g2146/smplx/" in outs and "dancecap/work/d_g2146/smplx_rf2_noslip/" in outs
    assert "dancecap/work/d_g2146/mamma/prepare.json" in outs
    assert job["priority"] == ch.estimate["total_min"] and job["max_minutes"] >= job["estimate"]["total_min"]
    assert job["cwd"] == "dancecap" and job["env"]["DANCECAP_FARM_DANCE"] == "d"
    json.dumps(job)                                                            # serialisable


def test_an_s1_chunk_only_runs_what_is_left_and_kinds_are_validated():
    spec = full_spec()
    ch = C.Chunk(id="b", k0=1027, k1=2312, core_k0=1148, core_k1=2252, cams=["01"], take="d_b", origin="s1", stages_done=["front"],
                 estimate={"front_min": 1.0, "fit_min": 50.0, "refine_min": 2.0, "total_min": 52.0})
    job = C.build_job(ch, spec, FakeWorld(), hash_inputs=False, code={})
    assert job["stages"] == ["fit", "refined"] and not any(s["name"].startswith("chunk_") for s in job["steps"])
    ch.stages_done = ["front", "fit"]
    assert C.build_job(ch, spec, FakeWorld(), kind="refine", hash_inputs=False, code={})["stages"] == ["refined"]
    with pytest.raises(ValueError):
        C.build_job(ch, spec, FakeWorld(), kind="bogus", hash_inputs=False, code={})
    assert C.stages_for(ch, "front") == []                                      # nothing to do: no steps


def test_prepared_archive_input_is_deterministic_and_content_addressed(tmp_path, monkeypatch):
    root = patch_root(monkeypatch, tmp_path)
    monkeypatch.setattr(C, "WORKSPACE", tmp_path)
    monkeypatch.setattr(W, "_CACHE", W.HashCache(tmp_path / "hc.json"))
    spec = full_spec()
    wk = root / "work" / "d_a"
    write_prepare(wk, -998, 100, ["01", "02"])
    seq = "d_a_c4"
    (wk / "mamma" / "ma_2d" / seq).mkdir(parents=True)
    (wk / "mamma" / "ma_2d" / seq / "cam01.npz").write_bytes(b"x" * 1000)
    (wk / "mamma" / "ma_cap" / seq / "gt").mkdir(parents=True)
    (wk / "mamma" / "ma_cap" / seq / "gt" / "cam01.npz").write_bytes(b"y" * 500)
    ch = C.Chunk(id="a", k0=-998, k1=119, core_k0=-998, core_k1=-1, take="d_a", origin="s1", stages_done=["front"])
    a = C.prepared_input(spec, ch)
    assert a["archive"] and a["path"] == "dancecap/work/d_a/mamma/prepared.tar" and a["n_files"] == 4
    import hashlib
    import tarfile
    tar = tmp_path / a["local"]
    assert hashlib.sha256(tar.read_bytes()).hexdigest() == a["sha256"] and tar.stat().st_size == a["bytes"]
    with tarfile.open(tar) as t:
        assert sorted(t.getnames()) == ["dancecap/work/d_a/mamma/ma3d_config.yaml", "dancecap/work/d_a/mamma/ma_2d/d_a_c4/cam01.npz",
                                        "dancecap/work/d_a/mamma/ma_cap/d_a_c4/gt/cam01.npz", "dancecap/work/d_a/mamma/prepare.json"]
    import os
    import time
    for p in wk.rglob("*"):
        if p.is_file():
            os.utime(p, (time.time() + 5, time.time() + 5))                     # new mtimes, same bytes
    monkeypatch.setattr(W, "_CACHE", W.HashCache(tmp_path / "hc2.json"))
    b = C.prepared_input(spec, ch)
    assert b["sha256"] == a["sha256"] and b["tree_sha256"] == a["tree_sha256"]
    (wk / "mamma" / "ma_2d" / seq / "cam01.npz").write_bytes(b"z" * 1000)
    c = C.prepared_input(spec, ch)
    assert c["sha256"] != a["sha256"] and c["tree_sha256"] != a["tree_sha256"]


def test_chunk_take_toml_parses_and_pins_the_world(monkeypatch):
    spec = full_spec()
    spec.cfg["cameras"] = {"01": {"mount": "handheld", "zoom": False, "source": "user"}}
    job = {"world_id": "w" * 12, "pose_id": "p" * 12, "cams": ["01", "02"],
           "chunk": {"id": "g2146", "take": "d_g2146", "k0": 1028, "k1": 2141, "core_k0": 1148, "core_k1": 2021, "core_s": 29.1}}
    monkeypatch.delenv("HM_ROOT", raising=False)
    monkeypatch.delenv("HM_WORLD_DIR", raising=False)
    cfg = tomllib.loads(C.chunk_take_toml(job, spec))
    assert cfg["name"] == "d_g2146" and cfg["segment"]["fit_k"] == [1028, 2141] and cfg["segment"]["cams"] == ["01", "02"]
    assert cfg["segment"]["grid_anchor"] == pytest.approx(30.151176844227862) and cfg["segment"]["parent"] == "d"
    assert cfg["exclude"] == ["03", "04", "05", "06", "07", "09"] and cfg["ingest"]["max_fps"] == 30
    assert cfg["world"] == {"dance": "d", "id": "w" * 12, "pose_id": "p" * 12, "role": "chunk"}
    w0, w1 = cfg["window"]
    assert w0 == pytest.approx(FULL.t(1028) - 2.0) and w1 == pytest.approx(FULL.t(2142) + 2.0)
    assert cfg["source_dir"] == "../../dances/Cap" and cfg["cameras"]["01"]["mount"] == "handheld"
    monkeypatch.setenv("HM_ROOT", "/workspace/hm")                              # on a pod: videos straight from the volume
    cfg = tomllib.loads(C.chunk_take_toml(job, spec))
    assert cfg["source_dir"] == "/workspace/hm/src/Cap"
