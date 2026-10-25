"""Unit tests of the two-camera farm profiles (farm/ca_profile.py: carlos_aline and lucas_luiza), `farm plan --final-only`, recap_front's HM_FRONT_MASTER and the
two-view share of dancecap.recap_fitreport (LL2V, runbook steps 12, 13, 15 and 16).

Synthetic teacher tracks, offsets files and prepared chunk files in temp dirs: no footage, no pod, no GPU (worker.gpu_info is stubbed). The carlos_aline regression
gate of the runbook (`farm --dance carlos_aline plan --dry-run --json` identical before / after) cannot run on this PC (no work/carlos_aline): the "before" is the
original module from git, run on the same fixtures as the refactored one.

    ../.venv/bin/python -m pytest tests/test_farm_twoview_profile.py -q      (from dancecap/)
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dancecap.farm as farm  # noqa: E402
from dancecap import recap_fitreport as FR  # noqa: E402
from dancecap import recap_front as rf  # noqa: E402
from dancecap.farm import analysis as A  # noqa: E402
from dancecap.farm import ca_profile as cap  # noqa: E402
from dancecap.farm import chunks as C  # noqa: E402
from dancecap.farm import cli as farm_cli  # noqa: E402
from dancecap.farm import watch as Wt  # noqa: E402
from dancecap.farm.state import State  # noqa: E402
from dancecap.farm.timeline import DanceSpec, Grid  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CA_ORIG_SHA = "6e0afee5fbf9aadfe639a9ada8a560f3976b7730"      # the last commit of the carlos_aline-only ca_profile.py (module constants, before TwoView)


@pytest.fixture(autouse=True)
def _no_gpu_query(monkeypatch):
    """validate_plan builds a farm Worker, which asks nvidia-smi for the GPU: never here."""
    from dancecap.farm import worker
    monkeypatch.setattr(worker, "gpu_info", lambda: {})


def teacher(t: np.ndarray, *, low_every: int = 0, clinch: tuple | None = None) -> dict:
    """One camera's teacher arrays: both dancers confident, 1.5 m apart on the floor; every low_every-th block of frames the follow is unsure (edge crops); inside
    clinch = (t0, t1) the couple is merged and 0.4 m apart."""
    n = len(t)
    conf = np.full((n, 2), 0.9)
    if low_every:
        conf[(np.arange(n) % low_every) < low_every // 5, 1] = 0.2
    floor = np.zeros((n, 2, 2))
    floor[:, 1, 0] = 1.5
    merged = np.zeros((n, 2), bool)
    if clinch:
        m = (t >= clinch[0]) & (t < clinch[1])
        floor[m, 1, 0] = 0.4
        merged[m] = True
    return {"t": t, "ok": np.ones(n, bool), "conf": conf, "merged": merged, "floor": floor}


def dumps(o) -> str:
    """JSON with the key order kept (byte-identical means the same order too), without the `created` stamps of two runs."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k != "created"}
        if isinstance(x, (list, tuple)):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(o), default=str)


def public(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if not k.startswith("_")}


DANCERS = {"lead": {"name": "A", "pid": 1, "stature_m": 1.75, "shoe_sole_m": 0.025, "description": "grey t-shirt"},
           "follow": {"name": "B", "pid": 2, "stature_m": 1.63, "shoe_sole_m": 0.06, "description": "white top"}}


# ------------------------------------------------------------------------------------------------ carlos_aline: identical to the original module


CA_SPANS = {"01": (3.6286, 202.08), "02": (14.8158, 201.0)}


def ca_spec() -> DanceSpec:
    cfg = {"name": "carlos_aline", "source_dir": "../../dances/Carlos-Aline", "alignment": "../../prototype/larissa_kadu/alignment_carlos_aline.json",
           "window": [3.62, 197.93], "chunks": {"chunk_s": 60.0, "overlap_s": 4.0, "mode": "windows", "profile": "carlos_aline"},
           "stitch": {"mono": [[3.6, 14.82]]}, "dance": {"level": "professional", "category": "jack_and_jill"}, "dancers": DANCERS,
           "cameras": {"01": {"mount": "tripod", "zoom": True}, "02": {"mount": "tripod", "zoom": False}}}
    return DanceSpec(name="carlos_aline", grid=Grid(0.0, 109, 5937), cams=["01", "02"], cfg=cfg)


def ca_teachers() -> dict:
    return {"01": teacher(np.arange(3.6286, 202.08, 1 / 23.976), low_every=40, clinch=(120.0, 150.0)),
            "02": teacher(np.arange(14.8158, 201.0, 1 / 30.0), low_every=70, clinch=(120.0, 150.0))}


def load_original(tmp_path: Path):
    """ca_profile.py as it was before the TwoView refactor, imported from git as a sibling module of the farm package (its relative imports resolve)."""
    try:
        src = subprocess.run(["git", "show", f"{CA_ORIG_SHA}:dancecap/dancecap/farm/ca_profile.py"], cwd=REPO, capture_output=True, text=True, timeout=60,
                             check=True).stdout
    except (OSError, subprocess.SubprocessError) as e:          # the regression gate must not pass silently (a shallow clone has no such commit)
        pytest.fail(f"the carlos_aline regression gate needs commit {CA_ORIG_SHA[:7]} (the original farm/ca_profile.py) in the git history: {e}")
    p = tmp_path / "ca_profile_orig.py"
    p.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("dancecap.farm._ca_profile_orig", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert not hasattr(mod, "TwoView") and mod.FIT_S_PER_FRAME_2CAM == 0.9
    return mod


@pytest.fixture
def ca_env(tmp_path, monkeypatch):
    old = load_original(tmp_path)
    tt = ca_teachers()
    monkeypatch.setattr(old, "load_teachers", lambda cams=old.CAMS: tt)
    monkeypatch.setattr(cap, "load_teachers", lambda P=cap.CA: tt)
    monkeypatch.setattr(C, "camera_spans", lambda spec, refresh=False: dict(CA_SPANS))
    monkeypatch.setattr(A, "_local_motion", lambda spec, chunk: False)      # hermetic: chunks pulled into the real work dir would travel as inputs
    return old, tt, ca_spec()


@pytest.mark.parametrize("claim_second", [False, True])
def test_ca_plan_job_specs_and_validation_are_identical_to_the_original(ca_env, claim_second, capsys):
    old, tt, spec = ca_env
    claimed = set()
    if claim_second:                                                  # cmd_plan passes queued / running chunk ids, with and without the prefix
        cid = cap.make_plan(spec, hash_inputs=False, code={})["_chunks"][1].id
        claimed = {cid, cid[len("ca"):]}
    a = old.make_plan(spec, hash_inputs=False, code={}, claimed_ids=set(claimed))
    b = cap.make_plan(spec, hash_inputs=False, code={}, claimed_ids=set(claimed))
    assert len(b["_jobs"]) == len(a["_jobs"]) >= 4 and [j["id"] for j in b["_jobs"]][0] == "cashape"
    for ja, jb in zip(a["_jobs"], b["_jobs"]):
        assert dumps(ja) == dumps(jb), ja["id"]
    assert dumps(public(a)) == dumps(public(b))                                         # what `plan --dry-run --json` prints, validation aside
    assert dumps(old.validate_plan(a, spec)) == dumps(cap.validate_plan(b, spec))       # ... and the validation
    old.print_plan(a)
    text_a = capsys.readouterr().out
    cap.print_plan(b)
    assert capsys.readouterr().out == text_a
    assert all("owner" not in j for j in b["_jobs"]) and all("HM_FRONT_MASTER" not in j["env"] for j in b["_jobs"])


def test_ca_estimates_shape_window_and_constants_are_the_original(ca_env):
    old, tt, spec = ca_env
    ch = C.Chunk(id="cax1408", k0=1408, k1=1768, core_k0=1408, core_k1=1768, overlap_k=0, cams=["01", "02"], take="carlos_aline_cax1408")
    assert old.estimate_chunk(ch, cam_frames=2 * ch.T) == cap.estimate_chunk(ch, cam_frames=2 * ch.T)       # ca_fix.estimate calls it with the default profile
    assert old.estimate_chunk(ch, shape=True) == cap.estimate_chunk(ch, shape=True)
    assert old.shape_window(tt) == cap.shape_window(tt)
    assert old.window_cams(tt, 30.0, 80.0) == cap.window_cams(tt, 30.0, 80.0)
    P = cap.CA
    assert (P.shape_take, P.in_tar_rel, P.pin_rel, P.job_env, P.sync_fixed) == (old.SHAPE_TAKE, old.IN_TAR_REL, old.PIN_REL, old.JOB_ENV, old.SYNC_FIXED)
    assert list(P.job_env) == list(old.JOB_ENV) and P.master == "01" and cap.PROFILES["carlos_aline"] is P


def test_ca_input_tar_has_the_original_members(tmp_path, monkeypatch, ca_env):
    old = ca_env[0]
    work = tmp_path / "work" / "carlos_aline"
    for d in ("cameras", "people/pose_tables"):
        (work / d).mkdir(parents=True)
    rng = np.random.default_rng(3)
    for c in ("01", "02"):
        np.savez(work / "cameras" / f"{c}.npz", t=np.arange(5.0), R=rng.normal(size=(5, 3, 3)))
        np.savez(work / "people" / f"teachers_{c}.npz", t=np.arange(5.0), ok=np.ones(5, bool))
        np.savez(work / "people" / "pose_tables" / f"{c}.npz", kp=rng.normal(size=(5, 2, 17, 3)), n=np.array([2, 2, 1, 0, 2]), det_conf=np.ones((5, 2)))
    (work / "cameras" / "recap_cams.json").write_text("{}")
    (work / "cameras" / "world_from_sfm.json").write_text('{"s": 1.0}')
    align = tmp_path / "alignment_carlos_aline.json"
    align.write_text('{"reference": "01"}')
    monkeypatch.setattr(old, "WORK", work)
    monkeypatch.setattr(old, "PEOPLE", work / "people")
    monkeypatch.setattr(old, "ALIGNMENT", align)
    monkeypatch.setattr(cap, "WORK_ROOT", tmp_path / "work")
    ra = old.build_in(tmp_path / "a.tar")
    slim_a = {c: dict(np.load(work / "people" / "in" / "pose_tables" / f"{c}.npz")) for c in ("01", "02")}
    rb = cap.build_in(tmp_path / "b.tar", P=dataclasses.replace(cap.CA, alignment=align))
    for k in ("path", "n_files", "cameras", "archive"):
        assert ra[k] == rb[k], k
    with tarfile.open(tmp_path / "a.tar") as ta, tarfile.open(tmp_path / "b.tar") as tb:
        assert ta.getnames() == tb.getnames() and "prototype/larissa_kadu/alignment_carlos_aline.json" in tb.getnames()
        for m in ta.getmembers():
            if "/people/pose_tables/" in m.name:     # re-written npz (zip timestamps): same arrays
                continue
            assert ta.extractfile(m).read() == tb.extractfile(m.name).read(), m.name
    for c in ("01", "02"):
        z = np.load(work / "people" / "in" / "pose_tables" / f"{c}.npz")
        assert set(z.files) == {"kp", "n"} and np.array_equal(z["kp"], slim_a[c]["kp"]) and np.array_equal(z["n"], slim_a[c]["n"])


# ------------------------------------------------------------------------------------------------ lucas_luiza: the two-view profile


LL_SPANS = {"01": (12.1229, 228.6), "02": (16.8, 213.5)}
DELTA, RATE, T_REF = -0.0423, 2e-5, 115.15


def ll_spec(refined_dir: str = "smplx_rf2_noslip") -> DanceSpec:
    cfg = {"name": "lucas_luiza", "source_dir": "../../../../Videos/Zouk/lucas-luisa-jnj", "alignment": "../work/lucas_luiza/clock/alignment.json",
           "window": [16.8, 213.5], "timeline": {"anchor": 0.0, "hz": 30.0}, "chunks": {"profile": "lucas_luiza", "mode": "windows", "chunk_s": 60.0, "overlap_s": 4.0},
           "sync": {"required": True}, "dance": {"level": "professional", "category": "jack_and_jill"}, "dancers": DANCERS, "export": {"smplx_dir": refined_dir},
           "cameras": {"01": {"mount": "handheld", "zoom": False}, "02": {"mount": "tripod", "zoom": True}}}
    return DanceSpec(name="lucas_luiza", grid=Grid(0.0, 504, 6405), cams=["01", "02"], cfg=cfg, refined_dir=refined_dir)


def ll_teachers(t01_start: float = 16.79) -> dict:
    """01 = the Steadicam (29 fps, edge crops: the follow is unsure in 20 % of its frames), 02 = the tripod (every 2nd frame of 59.94 fps); a clinch at 60-90 s."""
    return {"01": teacher(t01_start + np.arange(int((213.56 - t01_start) * 29)) / 29.0, low_every=50, clinch=(60.0, 90.0)),
            "02": teacher(16.800117 + np.arange(5896) * 2 / 59.94, clinch=(60.0, 90.0))}


def write_offsets(root: Path, **o01) -> dict:
    doc = {"version": 1, "method": "two_view_epipolar", "gauge": "camera 02 = 0",
           "offsets": {"01": {"delta": DELTA, "delta_ms": 1000 * DELTA, "stable": True, "use": True, "rate": RATE, "use_rate": True, "t_ref": T_REF,
                              "at_search_edge": False, "spread_ms": 4.0, **o01},
                       "02": {"delta": 0.0, "delta_ms": 0.0, "stable": True, "use": True}}}
    p = root / "lucas_luiza" / "sync" / "offsets.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc))
    return doc


@pytest.fixture
def ll_env(tmp_path, monkeypatch):
    root = tmp_path / "dancecap"
    (root / "work").mkdir(parents=True)
    monkeypatch.setattr(farm, "PACKAGE_ROOT", root)                 # farm_dir / jobs_dir / state.json under the temp tree
    monkeypatch.setattr(cap, "WORK_ROOT", root / "work")
    write_offsets(root / "work")
    tt = ll_teachers()
    monkeypatch.setattr(cap, "load_teachers", lambda P=cap.CA: tt)
    monkeypatch.setattr(C, "camera_spans", lambda spec, refresh=False: dict(LL_SPANS))
    monkeypatch.setattr(A, "_local_motion", lambda spec, chunk: False)
    monkeypatch.delenv("HM_OWNER", raising=False)
    for k in ("HM_QUEUE_OWNER", "HM_POD_OWNER"):                     # runbook step 13's exports (launch_env_claims_the_owner)
        monkeypatch.setenv(k, "LL2V")
    return root, tt, ll_spec()


def quiet_inputs(monkeypatch):
    """A non-dry `plan` without hashing videos or packing ll_in.tar."""
    monkeypatch.setattr(C, "source_inputs", lambda spec, cams: ([], {}))
    monkeypatch.setattr(cap, "build_in", lambda out=None, P=cap.CA: {"path": P.in_tar_rel, "archive": True, "sha256": "0" * 64, "bytes": 1})
    monkeypatch.setattr(C, "code_info", lambda *x, **k: {})


def plan_args(**kw) -> argparse.Namespace:
    return argparse.Namespace(**{"dance": "lucas_luiza", "dry_run": False, "json": False, "out": None, "final_only": False, **kw})


def expected_ms(chunk: dict) -> float:
    t_mid = 0.5 * (chunk["k0"] / 30.0 + (chunk["k1"] + 1) / 30.0)
    return round(1000 * (DELTA + RATE * (t_mid - T_REF)), 2)


def test_ll_plan_forces_both_cameras_owner_env_and_the_measured_offsets(ll_env):
    root, tt, spec = ll_env
    P = cap.LL
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=P)
    jobs = {j["id"]: j for j in plan["_jobs"]}
    chunks = [j for j in plan["_jobs"] if j.get("what") == "chunk"]
    assert [c.id for c in plan["_chunks"]] == [j["id"] for j in chunks] and len(chunks) == 4 and all(j["id"].startswith("llg") for j in chunks)
    assert all(1565 <= j["chunk"]["T"] <= 1566 for j in chunks) and plan["shape"]["id"] == "llshape" and "llfinal" in jobs
    for j in plan["_jobs"]:                                          # shape, chunks AND the final job
        assert j["owner"] == "LL2V" and j["env"]["HM_FRONT_MASTER"] == "02" and j["env"]["HM_TEACHER_EVIDENCE"] == "1", j["id"]
        assert j["env"]["HM_FRONT_DANCE"] == "lucas_luiza" and j["env"]["HM_FRONT_CAPTURE"] == "lucas-luisa-jnj" and j["env"]["HM_SHAPE_JOB"] == "llshape"
        assert not any(k.startswith("MAMMA_MONO") for k in j["env"])
    for j in chunks + [jobs["llshape"]]:
        assert j["cams"] == ["01", "02"]
        sf = j["chunk"]["sync_fixed"]
        assert sf["no_measure"] and sf["offsets_ms"]["02"] == 0.0 and sf["offsets_ms"]["01"] == pytest.approx(expected_ms(j["chunk"]), abs=1e-9)
    assert len({j["chunk"]["sync_fixed"]["offsets_ms"]["01"] for j in chunks}) == 4           # delta(t) at each chunk's own centre
    assert all(r["monocular_until_s"] is None and "cams_rule" in r for r in plan["chunks"])
    assert plan["owner"] == "LL2V" and plan["master"] == "02" and set(plan["sync"]["offsets_ms"]) == {j["id"] for j in chunks} | {"llshape"}
    st = plan["shape"]["stats"]
    assert st["both_cameras_both_dancers"] >= 0.6 and st["apart"] >= 0.5 and not (60.0 - 30.0 < plan["shape"]["window"][0] < 90.0)
    assert spec.grid.t_start + 1.0 <= plan["shape"]["window"][0] and plan["shape"]["window"][1] <= spec.grid.t_end - 1.0


def test_ll_chunk_steps_add_the_slip_variant_and_the_shape_job_checks_the_stature(ll_env):
    spec = ll_env[2]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    ch = next(j for j in plan["_jobs"] if j.get("what") == "chunk")
    names = [s["name"] for s in ch["steps"]]
    assert names == ["chunk_take", "front", "qa_mask", "await_pin", "fit", "refine", "quality", "refine_slip", "quality_slip", "evidence"]
    st = {s["name"]: s for s in ch["steps"]}
    assert st["refine_slip"]["cmd"].startswith(f"python -m dancecap.pose_refine {ch['take']} --src smplx --dst smplx_rf2_slip --mode refine")
    assert st["refine_slip"]["cmd"].endswith("--device cuda --w-slip 0.5 --slip-max-v 0.15") and "--w-slip 0.0" in st["refine_slip"]["cmd"]   # argparse keeps the last
    assert st["refine_slip"]["save"] == [f"dancecap/work/{ch['take']}/smplx_rf2_slip/"] and not st["refine_slip"]["optional"]
    assert st["quality_slip"]["cmd"].endswith("--smplx-dir smplx_rf2_slip --tag rf2_slip") and st["quality_slip"]["optional"]
    assert st["refine"]["cmd"].split(" --dst ")[1].startswith("smplx_rf2_noslip ") and st["refine_slip"]["stage"] == "refined"
    assert {"path": f"dancecap/work/{ch['take']}/smplx_rf2_slip/"} in ch["outputs"]
    e = ch["estimate"]
    assert e["fit_min"] == pytest.approx((74 + 1.9 * ch["chunk"]["T"]) / 60, abs=0.06) and e["variants_min"] == pytest.approx(e["refine_min"] + e["quality_min"], abs=0.1)
    assert st["fit"]["timeout_min"] >= 1.6 * e["fit_min"] and st["fit"]["env"]["MAMMA_INIT_BETAS"].endswith("lucas_luiza_llshape/mamma_pin_betas.json")
    sh = next(j for j in plan["_jobs"] if j["id"] == "llshape")
    assert [s["name"] for s in sh["steps"]] == ["chunk_take", "front", "qa_mask", "fit", "pin", "scale_check", "evidence"]
    sc = next(s for s in sh["steps"] if s["name"] == "scale_check")
    assert sc["optional"] and sc["cmd"].startswith("python -m dancecap.cameras.world_scale check lucas_luiza --smplx-dir work/lucas_luiza_llshape/smplx --strict")
    assert sc["cmd"].endswith("> work/lucas_luiza_llshape/scale_check.log 2>&1") and "variants_min" not in sh["estimate"]
    assert {"path": "dancecap/work/lucas_luiza_llshape/scale_check.json", "optional": True} in sh["outputs"]
    assert sh["inputs"] == [{"path": "src/lucas-luisa-jnj/ll_in.tar", "archive": True}]


def test_ll_validate_plan_passes_and_carries_every_two_view_gate(ll_env, monkeypatch):
    spec = ll_env[2]
    monkeypatch.delenv("HM_OWNER", raising=False)
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    v = cap.validate_plan(plan, spec, cap.LL)
    assert v["ok"], {k: c for k, c in v["checks"].items() if not c["ok"]}
    for k in ("owner_set", "launch_env_claims_the_owner", "no_MAMMA_MONO", "env_has_teacher_evidence", "env_has_the_master_camera", "sync_fixed_is_the_measured_offset",
              "both_cams_cover_window_ends", "every_fit_job_has_both_cameras", "shape_window_sees_both", "fit_timeouts_cover_the_estimate", "plan_cost_within",
              "budget_covers_the_fleet", "final_stitches_every_chunk", "worker_queue_and_claim_rules", "capture_is_the_upload_folder"):
        assert v["checks"][k]["ok"], k
    assert "HM_OWNER" not in os.environ                               # the owner was set only for the claim-rule check
    b = plan["budget"]
    assert b["usd"] == 14.0 and b["fleet_worst_usd"] <= 14.0 and b["gpu_jobs"] == 5 and b["gpus"] == ["NVIDIA GeForce RTX 4090"]
    assert b["hard_minutes"] >= max(j["max_minutes"] for j in plan["_jobs"] if j["needs_gpu"]) and plan["costs"]["usd"] <= 6.0
    gmax = max(j["max_minutes"] for j in plan["_jobs"] if j["needs_gpu"])
    assert b["hard_minutes"] == int(np.ceil(gmax + 15)) and b["worker_job_limit_min"] == round(1.25 * gmax + 5, 1)     # the pod ceiling is the tighter cap (documented)
    assert b["fits_up_to_4090_hr"] >= b["price_4090_hr"]
    assert v["checks"]["both_cams_cover_window_ends"]["cams"]["01"]["from"] == "people/teachers_01.npz t"          # no camera tracks in the fixture


@pytest.mark.parametrize("breakage, check", [
    (lambda p: p["_jobs"][1]["env"].pop("HM_FRONT_MASTER"), "env_has_the_master_camera"),
    (lambda p: p["_jobs"][-1]["env"].pop("HM_FRONT_MASTER"), "env_has_the_master_camera"),             # the final job too
    (lambda p: p["_jobs"][0]["env"].pop("HM_TEACHER_EVIDENCE"), "env_has_teacher_evidence"),
    (lambda p: p["_jobs"][2].pop("owner"), "owner_set"),
    (lambda p: p["_jobs"][2]["steps"][4].setdefault("env", {}).update(MAMMA_MONO="1"), "no_MAMMA_MONO"),
    (lambda p: p["_jobs"][1]["chunk"]["sync_fixed"]["offsets_ms"].update({"01": 0.0}), "sync_fixed_is_the_measured_offset"),
    (lambda p: p["_jobs"][1]["chunk"]["sync_fixed"]["offsets_ms"].pop("01"), "sync_fixed_is_the_measured_offset"),
    (lambda p: p["_jobs"][0]["chunk"]["sync_fixed"].update(no_measure=False), "sync_fixed_is_the_measured_offset"),
    (lambda p: p["_jobs"][3].update(cams=["02"]), "every_fit_job_has_both_cameras"),
    (lambda p: p["shape"]["stats"].update(apart=0.1), "shape_window_sees_both"),                     # LL: apart >= 0.2 (close embrace)
    (lambda p: p["shape"]["stats"].update(both_cameras_both_dancers=0.3), "shape_window_sees_both"),  # LL: >= 0.33 (the Steadicam's ok frames)
    (lambda p: p["costs"].update(usd=7.5), "plan_cost_within"),
    (lambda p: p["_jobs"][-1].update(after=[]), "final_stitches_every_chunk"),                      # what a re-plan with claimed chunks would build
    (lambda p: p["_jobs"][-1]["stitch_chunks"].pop(1), "final_stitches_every_chunk"),
    (lambda p: p["budget"].update(fleet_worst_usd=14.5), "budget_covers_the_fleet"),
])
def test_ll_strict_checks_fail_on_each_breakage(ll_env, breakage, check):
    spec = ll_env[2]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    assert cap.strict_checks(plan, spec, cap.LL)[check]["ok"]
    breakage(plan)
    assert not cap.strict_checks(plan, spec, cap.LL)[check]["ok"]


def test_ll_validate_plan_fails_without_the_master_env_and_with_an_unmeasured_offset(ll_env):
    spec = ll_env[2]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    plan["_jobs"][2]["env"].pop("HM_FRONT_MASTER")
    v = cap.validate_plan(plan, spec, cap.LL)
    assert not v["ok"] and not v["checks"]["env_has_the_master_camera"]["ok"] and plan["_jobs"][2]["id"] in v["checks"]["env_has_the_master_camera"]["missing"]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    (ll_env[0] / "work" / "lucas_luiza" / "sync" / "offsets.json").unlink()            # the plan's offsets cannot be checked against the measurement any more
    v = cap.validate_plan(plan, spec, cap.LL)
    assert not v["ok"] and "missing" in v["checks"]["sync_fixed_is_the_measured_offset"]["error"]


def test_ll_window_ends_must_be_covered_by_both_cameras_after_the_offset(ll_env, monkeypatch):
    root, _, spec = ll_env
    late = ll_teachers(t01_start=16.85)                               # the Steadicam's first window frame + delta lands after the grid's first instant + 1 frame
    monkeypatch.setattr(cap, "load_teachers", lambda P=cap.CA: late)
    write_offsets(root / "work", delta=0.040, delta_ms=40.0, rate=0.0)
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    chk = cap.strict_checks(plan, spec, cap.LL)["both_cams_cover_window_ends"]
    assert not chk["ok"] and not chk["cams"]["01"]["ok"] and chk["cams"]["02"]["ok"]
    assert chk["both_cover_s"][0] == pytest.approx(16.89, abs=1e-3) and "narrow the take's window to [16.89," in chk["advice"]


def test_ll_window_ends_fail_at_the_end_when_the_offset_is_negative(ll_env):
    """Camera 01's last frame 213.514 s - 60 ms lands more than one frame (1/29 s) before the grid's last instant 213.5 s: the side a negative delta_01 breaks."""
    root, _, spec = ll_env
    write_offsets(root / "work", delta=-0.060, delta_ms=-60.0, rate=0.0)
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    chk = cap.strict_checks(plan, spec, cap.LL)["both_cams_cover_window_ends"]
    c01 = chk["cams"]["01"]
    assert not chk["ok"] and not c01["ok"] and chk["cams"]["02"]["ok"] and c01["first_s"] <= 16.8 and c01["last_s"] < 213.5 - c01["frame_s"]
    assert chk["both_cover_s"][1] == pytest.approx(c01["last_s"]) and chk["advice"].startswith("narrow the take's window to [16.81, 213.45]")
    write_offsets(root / "work", delta=-0.040, delta_ms=-40.0, rate=0.0)       # within one frame: covered
    assert cap.strict_checks(cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL), spec, cap.LL)["both_cams_cover_window_ends"]["ok"]


def test_ll_window_ends_read_the_camera_tracks_registered_frames_when_they_are_here(ll_env):
    """mamma prepare's coverage is the camera track's ok frames (cameras/<cam>.npz t[ok]), not the teacher track's: a track that registers only from 17.5 s on
    leaves the first 0.7 s to camera 02."""
    root, tt, spec = ll_env
    cams = root / "work" / "lucas_luiza" / "cameras"
    cams.mkdir(parents=True)
    t = tt["01"]["t"]
    np.savez(cams / "01.npz", t=t, ok=t >= 17.5, R=np.zeros((len(t), 3, 3)))
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    chk = cap.strict_checks(plan, spec, cap.LL)["both_cams_cover_window_ends"]
    assert not chk["ok"] and chk["cams"]["01"]["from"] == "cameras/01.npz t[ok]" and chk["cams"]["02"]["from"] == "people/teachers_02.npz t"
    assert chk["cams"]["01"]["first_s"] == pytest.approx(float(t[t >= 17.5][0]) + DELTA + RATE * (float(t[t >= 17.5][0]) - T_REF), abs=1e-4)
    times = cap.coverage_times(tt, cap.LL)
    assert set(times) == {"01", "02"} and len(times["01"][0]) == int((t >= 17.5).sum())


@pytest.mark.parametrize("change, check", [
    (dict(fit_s_per_frame=0.9), "fit_timeouts_cover_the_estimate"),          # carlos_aline's rate on this take: below the measured two-camera rate
    (dict(budget_usd=10.0), "budget_covers_the_fleet"),                      # the route's worst case ($13.35) no longer fits
    (dict(capture="lucas-luiza"), "capture_is_the_upload_folder"),
])
def test_ll_profile_gates_fail_on_a_wrong_profile_value(ll_env, change, check):
    spec = ll_env[2]
    P = dataclasses.replace(cap.LL, **change)
    assert cap.strict_checks(cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL), spec, cap.LL)[check]["ok"]
    assert not cap.strict_checks(cap.make_plan(spec, hash_inputs=False, code={}, P=P), spec, P)[check]["ok"]


def test_ll_launch_env_must_claim_the_owner(ll_env, monkeypatch):
    spec = ll_env[2]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    for k, v in (("HM_QUEUE_OWNER", None), ("HM_POD_OWNER", None), ("HM_QUEUE_OWNER", "J4")):
        monkeypatch.setenv("HM_QUEUE_OWNER", "LL2V")
        monkeypatch.setenv("HM_POD_OWNER", "LL2V")
        monkeypatch.delenv(k) if v is None else monkeypatch.setenv(k, v)
        chk = cap.validate_plan(plan, spec, cap.LL)["checks"]["launch_env_claims_the_owner"]
        assert not chk["ok"] and "HM_QUEUE_OWNER=LL2V" in chk["fix"], (k, v)


def test_ll_sync_for_refuses_offsets_that_were_not_measured_by_the_two_view_scan(ll_env, monkeypatch):
    root, _, spec = ll_env
    nan = float("nan")
    for o01, doc_patch in (({"stable": False}, {}), ({"use": False}, {}), ({"at_search_edge": True}, {}), ({}, {"method": "gcc_phat"}),
                           ({}, {"offsets": {"01": {"delta": -0.04, "stable": True}, "02": {"delta": 0.012}}}), ({"delta": nan}, {}), ({"rate": nan}, {}),
                           ({}, {"offsets": {"01": {"delta": -0.04, "stable": True, "use": True}, "02": {"delta": nan}}})):
        doc = write_offsets(root / "work", **o01)
        doc.update(doc_patch)
        (root / "work" / "lucas_luiza" / "sync" / "offsets.json").write_text(json.dumps(doc))
        with pytest.raises(cap.SyncError):
            cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    (root / "work" / "lucas_luiza" / "sync" / "offsets.json").unlink()
    with pytest.raises(cap.SyncError, match="missing"):
        cap.LL.offsets_doc()
    monkeypatch.setattr(C, "code_info", lambda *x, **k: {})
    assert cap.cmd_plan(plan_args(dry_run=True), spec, cap.LL) == 2      # refused cleanly, nothing planned
    write_offsets(root / "work", use_rate=False)                     # a constant offset when the scan did not adopt its drift (mamma_dyn.read_sync_offsets' rule)
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    assert {j["chunk"]["sync_fixed"]["offsets_ms"]["01"] for j in plan["_jobs"] if j.get("what") in ("chunk", "shape")} == {round(1000 * DELTA, 2)}
    assert "drift" not in plan["sync"]
    write_offsets(root / "work", fit={"a_ms": -41.9})
    assert cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)["sync"]["drift"] == {"01": {"delta_ms": -42.3, "fit_a_ms": -41.9, "t_ref": T_REF}}


def test_ll_offsets_measured_on_other_clocks_are_refused(ll_env, monkeypatch):
    """sync.offsets_check, the rule mamma_dyn.read_sync_offsets applies: a file whose clocks block differs from the frames' is stale; one without a clocks block
    cannot be verified, which a [sync] required take (lucas_luiza) refuses."""
    root, _, spec = ll_env
    work = root / "work" / "lucas_luiza"
    for cam, (s, o) in {"01": (0.9999163, 12.1229), "02": (1.0, 0.0)}.items():
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps({"slope": s, "offset": o, "frames": []}))
    p = work / "sync" / "offsets.json"
    doc = json.loads(p.read_text())
    doc["clocks"] = {"alignment": "x", "sources": {"01": [0.9999163, 12.1229], "02": [1.0, 0.0]}}
    p.write_text(json.dumps(doc))
    assert cap.LL.offsets_doc()["clocks"]                              # recorded on these clocks: ok
    doc["clocks"]["sources"]["01"] = [0.9999163, 12.1529]             # measured before a re-ingest moved camera 01's clock by 30 ms
    p.write_text(json.dumps(doc))
    with pytest.raises(cap.SyncError, match="stale"):
        cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    from dancecap import sync as S
    del doc["clocks"]
    p.write_text(json.dumps(doc))
    assert cap.LL.offsets_doc()                                        # unrecorded: the synthetic take does not require sync ...
    monkeypatch.setattr(S, "take_clocks", lambda work: {"alignment": "x", "required": True, "sources": {}})
    with pytest.raises(cap.SyncError, match="unrecorded"):            # ... lucas_luiza does ([sync] required = true)
        cap.LL.offsets_doc()


def test_ll_plan_writes_its_budget_into_the_farm_policy_and_keeps_a_lower_one(ll_env, monkeypatch, capsys):
    root, _, spec = ll_env
    quiet_inputs(monkeypatch)
    a = plan_args()
    assert cap.cmd_plan(a, spec, cap.LL) == 0
    plan = json.loads((root / "work" / "lucas_luiza" / "farm" / "plan.json").read_text())
    pol = Wt.Policy.load("lucas_luiza")
    assert plan["validation"]["ok"] and pol.budget_usd == 14.0 and pol.hard_minutes == plan["budget"]["hard_minutes"] and pol.minutes <= pol.hard_minutes
    assert pol.gpus == ["NVIDIA GeForce RTX 4090"]                    # what the budget is priced on (min_vram_gb 24 refuses an L4)
    assert {p.stem for p in (root / "work" / "lucas_luiza" / "farm" / "jobs").glob("*.json")} == {j["id"] for j in plan["costs"]["jobs"]} | {"llfinal"}
    out = capsys.readouterr().out
    assert "queue --only llshape" in out and "launch --pods 1 --gpus 4090" in out and "up --dry-run" not in out
    Wt.Policy.from_dict({"budget_usd": 5.0, "gpus": ["NVIDIA RTX A5000"]}, pol).save("lucas_luiza")
    assert cap.cmd_plan(a, spec, cap.LL) == 0
    pol = Wt.Policy.load("lucas_luiza")
    assert pol.budget_usd == 5.0 and pol.gpus == ["NVIDIA RTX A5000"]       # a stricter ceiling and a GPU list set by hand are kept


def test_ll_plan_writes_nothing_when_a_gate_fails(ll_env, monkeypatch, tmp_path):
    root, _, spec = ll_env
    quiet_inputs(monkeypatch)
    monkeypatch.delenv("HM_QUEUE_OWNER")                              # the shell did not export the owner
    out = tmp_path / "plan.json"
    assert cap.cmd_plan(plan_args(out=str(out)), spec, cap.LL) == 2
    farm_d = root / "work" / "lucas_luiza" / "farm"
    assert not (farm_d / "jobs").exists() and not (farm_d / "plan.json").exists() and not State("lucas_luiza").load()["jobs"]
    assert Wt.Policy.load("lucas_luiza").budget_usd is None
    assert not json.loads(out.read_text())["validation"]["checks"]["launch_env_claims_the_owner"]["ok"]      # --out still has the details
    assert cap.cmd_plan(plan_args(dry_run=True), spec, cap.LL) == 0     # a dry run only looks


def test_ll_replan_is_refused_while_chunks_are_claimed_and_the_ceiling_never_drops_below_a_live_job(ll_env, monkeypatch, tmp_path, capsys):
    root, _, spec = ll_env
    quiet_inputs(monkeypatch)
    assert cap.cmd_plan(plan_args(), spec, cap.LL) == 0
    jobs = root / "work" / "lucas_luiza" / "farm" / "jobs"
    files = {p.name: p.read_bytes() for p in jobs.glob("*.json")}
    st = State("lucas_luiza")
    ids = sorted(cid for cid in st.load()["chunks"] if cid.startswith("llg"))
    st.set_job("llshape", status="pulled")
    for cid in ids:                                                   # what `queue --only <the 4 chunks>` does
        st.set_chunk(cid, status="queued")
        st.set_job(cid, status="queued")
    before = st.load()
    pol0 = Wt.Policy.load("lucas_luiza")
    assert cap.cmd_plan(plan_args(), spec, cap.LL) == 2
    err = capsys.readouterr().err
    assert "llg" in err and "queued" in err and "--final-only" in err and "refused" in err
    after = st.load()
    assert {p.name: p.read_bytes() for p in jobs.glob("*.json")} == files and Wt.Policy.load("lucas_luiza") == pol0
    assert after["jobs"]["llshape"]["status"] == "pulled" and after["jobs"]["llfinal"] == before["jobs"]["llfinal"]
    out = tmp_path / "dry.json"
    assert cap.cmd_plan(plan_args(dry_run=True, out=str(out)), spec, cap.LL) == 0                    # a dry run shows what the re-plan would be
    v = json.loads(out.read_text())["validation"]
    assert not v["ok"] and not v["checks"]["final_stitches_every_chunk"]["ok"] and "dry run: shown only" in capsys.readouterr().err
    longest = max(json.loads((jobs / f"{c}.json").read_text())["max_minutes"] for c in ids)
    res = cap.save_budget("lucas_luiza", {"usd": 14.0, "hard_minutes": 110})                          # a budget sized without the queued chunks
    assert res["hard_minutes"] == int(np.ceil(longest + 15)) and Wt.Policy.load("lucas_luiza").hard_minutes >= longest


def test_ll_replan_after_the_shape_pass_only_says_it_resets_the_shape(ll_env, monkeypatch, capsys):
    spec = ll_env[2]
    quiet_inputs(monkeypatch)
    assert cap.cmd_plan(plan_args(), spec, cap.LL) == 0
    State("lucas_luiza").set_job("llshape", status="pulled")          # G9 failed: world_scale rescale, step 11 again, then a new plan and shape pass
    capsys.readouterr()
    assert cap.cmd_plan(plan_args(), spec, cap.LL) == 0
    assert "llshape was pulled: this plan sets it back to planned" in capsys.readouterr().out and State("lucas_luiza").load()["jobs"]["llshape"]["status"] == "planned"


def test_ll_two_variants_may_not_share_a_step_name(ll_env):
    spec = ll_env[2]
    P = dataclasses.replace(cap.LL, refine_variants=(("smplx_rf2_slip", "--w-slip 0.5"), ("smplx_rf3_slip", "--w-slip 1.0")))
    with pytest.raises(ValueError, match="duplicate step names"):
        cap.make_plan(spec, hash_inputs=False, code={}, P=P)


def test_ll_print_plan_counts_the_slip_variant_in_the_refine_column(ll_env, capsys):
    spec = ll_env[2]
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    cap.print_plan(plan, cap.LL)
    rows = {ln.split()[0]: ln.split() for ln in capsys.readouterr().out.splitlines() if ln.startswith("llg")}
    for r in plan["chunks"]:
        e = r["estimate"]
        assert float(rows[r["id"]][6]) == pytest.approx(e["refine_min"] + e["quality_min"] + e["variants_min"], abs=0.051)
        assert e["total_min_4090"] == pytest.approx(e["front_min"] + e["fit_min"] + e["refine_min"] + e["quality_min"] + e["variants_min"] + 3.0, abs=0.2)


def test_final_only_rebuilds_just_the_final_job_and_is_refused_while_it_is_queued(ll_env, monkeypatch):
    root, _, spec = ll_env
    quiet_inputs(monkeypatch)
    a = plan_args()
    assert cap.cmd_plan(a, spec, cap.LL) == 0
    jobs = root / "work" / "lucas_luiza" / "farm" / "jobs"
    others = {p.name: p.read_bytes() for p in jobs.glob("*.json") if p.stem != "llfinal"}
    first = json.loads((jobs / "llfinal.json").read_text())
    st = State("lucas_luiza")
    ids = sorted(cid for cid in st.load()["chunks"] if cid.startswith("llg"))
    st.set_job("llfinal", status="queued")
    a.final_only = True
    slip = ll_spec("smplx_rf2_slip")                                  # [export] smplx_dir = the refine variant (G11)
    assert cap.cmd_plan(a, slip, cap.LL) == 2 and json.loads((jobs / "llfinal.json").read_text()) == first
    for cid in ids:
        st.set_chunk(cid, status="pulled")
    st.set_job("llfinal", status="pulled")
    monkeypatch.setattr(A, "_local_motion", lambda spec, chunk: True)  # the chunks' motion is on this PC now: the job still takes the volume's result.tar files
    assert cap.cmd_plan(a, slip, cap.LL) == 0
    job = json.loads((jobs / "llfinal.json").read_text())
    assert job["after"] == ids and job["local_chunks"] == [] and [i["path"] for i in job["inputs"]] == [f"chunks/{c}/out/result.tar" for c in ids]
    assert {"path": "dancecap/work/lucas_luiza/smplx_rf2_slip/", "optional": True} in job["outputs"] and first["outputs"] != job["outputs"]
    assert job["owner"] == "LL2V" and job["env"]["HM_FRONT_MASTER"] == "02" and all(v["world_id"] is None for v in job["chunk_worlds"].values())
    assert st.load()["jobs"]["llfinal"]["status"] == "planned"
    assert {p.name: p.read_bytes() for p in jobs.glob("*.json") if p.stem != "llfinal"} == others          # nothing else was rewritten


def test_final_only_dry_run_typo_and_tiling_refusals(ll_env, monkeypatch, tmp_path):
    root, _, spec = ll_env
    quiet_inputs(monkeypatch)
    assert cap.cmd_plan(plan_args(), spec, cap.LL) == 0
    jobs = root / "work" / "lucas_luiza" / "farm" / "jobs"
    first = (jobs / "llfinal.json").read_bytes()
    st = State("lucas_luiza")
    st.set_job("llfinal", status="pulled")
    out = tmp_path / "final.json"
    assert cap.cmd_plan(plan_args(final_only=True, dry_run=True, out=str(out)), ll_spec("smplx_rf2_slip"), cap.LL) == 0
    assert json.loads(out.read_text())["id"] == "llfinal" and (jobs / "llfinal.json").read_bytes() == first and st.load()["jobs"]["llfinal"]["status"] == "pulled"
    assert cap.cmd_plan(plan_args(final_only=True), ll_spec("smplx_rf2_slipp"), cap.LL) == 2                    # a refined dir no chunk job writes
    ids = sorted(cid for cid in st.load()["chunks"] if cid.startswith("llg"))
    st.set_chunk(ids[1], status="cancelled")                          # a hole in the grid
    assert cap.cmd_plan(plan_args(final_only=True), spec, cap.LL) == 2 and (jobs / "llfinal.json").read_bytes() == first


def test_farm_cli_dispatches_every_two_view_profile_and_refuses_final_only_elsewhere(monkeypatch):
    seen = []
    monkeypatch.setattr(cap, "cmd_plan", lambda a, spec, P=cap.CA: seen.append((spec.name, P.profile)) or 0)
    monkeypatch.setattr(DanceSpec, "load", classmethod(lambda cls, name: ll_spec() if name == "lucas_luiza" else ca_spec()))
    ns = farm_cli.build_parser().parse_args(["plan", "--final-only", "--dry-run"])
    assert ns.final_only and ns.dry_run
    for dance in ("lucas_luiza", "carlos_aline"):
        ns.dance = dance
        assert farm_cli.cmd_plan(ns) == 0
    assert seen == [("lucas_luiza", "lucas_luiza"), ("carlos_aline", "carlos_aline")]
    monkeypatch.setattr(DanceSpec, "load", classmethod(lambda cls, name: DanceSpec(name=name, grid=Grid(0.0, 0, 99), cfg={"chunks": {"profile": "john_lula"}})))
    ns.dance = "john_lula"
    assert farm_cli.cmd_plan(ns) == 2
    monkeypatch.setattr(DanceSpec, "load", classmethod(lambda cls, name: DanceSpec(name=name, grid=Grid(0.0, 0, 99), cfg={})))
    assert farm_cli.cmd_plan(ns) == 2                                 # the demo's gap planner has no final-only either


# ------------------------------------------------------------------------------------------------ recap_front: the master camera from the job env


def test_recap_front_gauges_on_the_master_camera_of_the_job_env(tmp_path, monkeypatch):
    work = tmp_path / "lucas_luiza_llg0000"
    for cam, (s, o) in {"01": (0.9999163, 12.1229), "02": (1.0, 0.0)}.items():
        (work / "frames" / cam).mkdir(parents=True)
        (work / "frames" / cam / "frames.json").write_text(json.dumps({"slope": s, "offset": o, "frames": []}))
    offs = {"01": -42.3, "02": 0.0}
    orig = os.environ.get("HM_FRONT_MASTER")
    session_rf = dict(vars(rf))   # the module as the session imported it (functions bind MASTER as a default when defined)
    try:
        monkeypatch.setenv("HM_FRONT_MASTER", "02")
        importlib.reload(rf)
        assert rf.MASTER == "02"
        doc = rf.fixed_sync_doc(work, offs, "test", ["01", "02"])
        assert doc["gauge"].startswith("camera 02") and doc["offsets"]["01"]["delta"] == -0.0423 and doc["offsets"]["02"]["delta"] == 0.0
        monkeypatch.delenv("HM_FRONT_MASTER")
        importlib.reload(rf)
        assert rf.MASTER == "01"                                      # every other take: unchanged
        doc = rf.fixed_sync_doc(work, offs, "test", ["01", "02"])
        assert doc["offsets"]["01"]["delta"] == 0.0                   # what a job without HM_FRONT_MASTER would fit: the measured offset dropped
    finally:                                                          # the module as the session imported it
        if orig is None:
            monkeypatch.delenv("HM_FRONT_MASTER", raising=False)
        else:
            monkeypatch.setenv("HM_FRONT_MASTER", orig)
        # put the session's module back as it was; a reload here would re-read HM_FRONT_DANCE, which other modules set at import
        # time in a full run (jl_front: setdefault john_lula), and leave recap_front on the wrong take for every later test
        vars(rf).clear()
        vars(rf).update(session_rf)


# ------------------------------------------------------------------------------------------------ recap_fitreport: the two-view share


def quality(cams=("01", "02")) -> dict:
    whole = {c: {"lead": {"median": 4.0, "p90": 7.0}, "follow": {"median": 4.5, "p90": 8.0}} for c in cams}
    return {"frames": 100, "span_s": [16.8, 20.1], "reproj_px": {"whole": whole, "all_cams_per_frame_median": {"lead": {"median": 5.0, "p90": 7.0},
                                                                                                             "follow": {"median": 5.5, "p90": 8.0}}},
            "reproj_feet_px": {"all_cams_per_frame_median": {"lead": {"median": 6.0}, "follow": {"median": 6.5}}},
            "jitter_4p5_12hz_mm": {"lead": {"whole_rms": 0.5}, "follow": {"whole_rms": 0.7}},
            "floor": {"lead": {"frac_below_15mm": 0.0}, "follow": {"frac_below_15mm": 0.0}},
            "rest_height_m": {"lead": {"fit": 1.78, "rel_vs_target": 0.003}, "follow": {"fit": 1.69, "rel_vs_target": 0.0}}}


def chunk_take(root: Path, cid: str = "llg0000", fitted_01: float = -42.3, planned_01: float = -42.3, with_ma2d: bool = True) -> Path:
    """A pulled two-camera chunk: prepare.json (T = 100), the fitted set's ma_2d visibilities, qa_mask.json, the quality json and the job spec.
    Camera 01 (Steadicam): the lead enters in frames 0-79 (80-89 have only 10 visible landmarks), the follow in 0-59. Camera 02: both everywhere, but qa_mask zeroed
    the follow in 90-99."""
    T, seq = 100, f"lucas_luiza_{cid}_c2"
    w = root / "work" / f"lucas_luiza_{cid}"
    (w / "mamma" / "ma_2d" / seq).mkdir(parents=True)
    vis = {c: np.zeros((T, 2, 512), np.float32) for c in ("01", "02")}
    vis["01"][:80, 0, :40] = 1.0
    vis["01"][80:90, 0, :10] = 1.0
    vis["01"][:60, 1, :200] = 0.8
    vis["02"][:, :, :300] = 1.0
    vis["02"][90:, 1] = 0.0
    for c, v in vis.items():
        if with_ma2d:
            np.savez(w / "mamma" / "ma_2d" / seq / f"cam{c}.npz", landmarks=np.zeros((T, 2, 512, 3), np.float32), visibilities=v)
    prep = {"grid": {"t0": 16.8, "T": T, "hz": 30.0}, "sync": {"offsets_ms": {"01": fitted_01, "02": 0.0}},
            "sets": {"c2": {"seq": seq, "cams": ["01", "02"], "inputs": {"01": {"grid_cam_ok": 0.9, "grid_visible_frac": {"lead": 0.75, "follow": 0.55}},
                                                                         "02": {"grid_cam_ok": 1.0, "grid_visible_frac": {"lead": 0.59, "follow": 0.59}}}}}}
    (w / "mamma" / "prepare.json").write_text(json.dumps(prep))
    (w / "mamma" / "qa_mask.json").write_text(json.dumps({"cams": {"01": {"masked_frames": {"lead": 0, "follow": 0}, "suspect": []},
                                                                   "02": {"masked_frames": {"lead": 0, "follow": 10}, "suspect": []}}}))
    (w / "quality").mkdir()
    (w / "quality" / "quality_rf2_noslip.json").write_text(json.dumps(quality()))
    jd = root / "work" / "lucas_luiza" / "farm" / "jobs"
    jd.mkdir(parents=True, exist_ok=True)
    (jd / f"{cid}.json").write_text(json.dumps({"id": cid, "chunk": {"sync_fixed": {"no_measure": True, "offsets_ms": {"01": planned_01, "02": 0.0}}}}))
    return w


@pytest.fixture
def fr_ll(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(FR, "DANCE", "lucas_luiza")
    monkeypatch.setattr(FR, "TWO_VIEW_MIN_SHARE", 0.7)
    return tmp_path


def test_two_view_share_counts_only_the_frames_that_enter_the_fit(fr_ll):
    w = chunk_take(fr_ll)
    tw = FR.two_view_share(w)
    assert tw["tag"] == "c2" and tw["frames"] == 100
    assert tw["cams"]["01"]["share"] == {"lead": 0.8, "follow": 0.6} and tw["cams"]["02"]["share"] == {"lead": 1.0, "follow": 0.9}
    assert tw["both"] == {"lead": 0.8, "follow": 0.6} and tw["cams"]["02"]["qa_masked_share"]["follow"] == 0.1 and tw["cams"]["01"]["grid_cam_ok"] == 0.9
    assert tw["cams"]["01"]["source"] == "ma_2d"


def test_fit_report_fails_a_chunk_that_is_monocular_for_one_dancer_in_one_camera(fr_ll):
    chunk_take(fr_ll)
    rep = FR.report(["llg0000"])
    r = rep["chunks"]["llg0000"]
    assert rep["gates"]["two_view_share"] == 0.7 and r["state"] == "FAIL" and r["two_view"]["both"]["follow"] == 0.6
    assert r["failures"] == ["follow: camera 01 two-view share 0.60 < 0.7 (camera ok 0.9, qa_mask zeroed 0.00)",
                             "follow: both cameras enter only 0.60 of the frames < 0.7 (the triangulated frames)"]
    t = FR.table(rep).splitlines()
    assert t[0].endswith("two-view share lead / follow |") and t[2].endswith("| 0.80 / 0.60 (both 0.80 / 0.60) |")
    json.dumps(rep)


def test_fit_report_fails_a_camera_whose_dancer_qa_mask_mostly_zeroed(fr_ll):
    w = chunk_take(fr_ll)
    (w / "mamma" / "qa_mask.json").write_text(json.dumps({"cams": {"01": {"masked_frames": {"lead": 0, "follow": 0}},
                                                                   "02": {"masked_frames": {"lead": 25, "follow": 10}, "suspect": []}}}))
    bad = FR.report(["llg0000"])["chunks"]["llg0000"]["failures"]
    assert "lead: camera 02 qa_mask zeroed 0.25 of the frames > 0.1" in bad and not any("follow: camera 02 qa_mask" in b for b in bad)    # 10 % is allowed


def test_fit_report_catches_a_fit_that_dropped_the_planned_offset(fr_ll):
    chunk_take(fr_ll, fitted_01=0.0)                                  # what a job without HM_FRONT_MASTER = 02 fits
    bad = FR.report(["llg0000"])["chunks"]["llg0000"]["failures"]
    assert any(b.startswith("sync: camera 01 fitted with 0.0 ms, the job planned -42.30 ms") for b in bad)


def test_fit_report_fails_two_cameras_that_carry_different_frames(fr_ll):
    """Each camera carries 0.75 of the lead's frames, but camera 01 the first 75 and camera 02 the last 75: only half of them are triangulated."""
    w = chunk_take(fr_ll)
    seq = w / "mamma" / "ma_2d" / "lucas_luiza_llg0000_c2"
    for cam, sl in (("01", slice(0, 75)), ("02", slice(25, 100))):
        v = np.zeros((100, 2, 512), np.float32)
        v[sl, 0, :40] = 1.0
        v[:, 1, :40] = 1.0
        np.savez(seq / f"cam{cam}.npz", landmarks=np.zeros((100, 2, 512, 3), np.float32), visibilities=v)
    (w / "mamma" / "qa_mask.json").write_text(json.dumps({"cams": {}}))
    r = FR.report(["llg0000"])["chunks"]["llg0000"]
    assert r["two_view"]["cams"]["01"]["share"]["lead"] == 0.75 and r["two_view"]["cams"]["02"]["share"]["lead"] == 0.75
    assert r["failures"] == ["lead: both cameras enter only 0.50 of the frames < 0.7 (the triangulated frames)"]


def test_fit_report_fails_a_camera_that_qa_mask_judged_suspect(fr_ll):
    """qa_mask masks nothing in a camera where both dancers are off the teacher in most frames (a clock / camera error): its share stays high, the gate fails it."""
    w = chunk_take(fr_ll)
    (w / "mamma" / "qa_mask.json").write_text(json.dumps({"cams": {"01": {"masked_frames": {"lead": 0, "follow": 0}, "suspect": []},
                                                                   "02": {"masked_frames": {"lead": 0, "follow": 0}, "suspect": ["lead", "follow"]}}}))
    bad = FR.report(["llg0000"])["chunks"]["llg0000"]["failures"]
    assert "camera 02: qa_mask judged both dancers off the teacher (suspect, nothing masked): check the video offset / the camera track" in bad
    assert not any(b.startswith("camera 01:") for b in bad)


def test_two_view_share_is_unknown_without_ma_2d_and_a_missing_front_fails(fr_ll):
    w = chunk_take(fr_ll, with_ma2d=False)
    tw = FR.two_view_share(w)
    assert tw["cams"]["02"]["share"] == {"lead": None, "follow": None} and tw["both"] == {} and tw["cams"]["01"]["source"] == "no ma_2d pulled"
    assert tw["cams"]["01"]["grid_visible_frac"] == {"lead": 0.75, "follow": 0.55}           # kept for reading, never compared with the frame-share gate
    bad = FR.two_view_failures(tw)
    assert "lead: camera 02 two-view share unknown (no ma_2d pulled)" in bad and "follow: both cameras' share unknown (no ma_2d pulled)" in bad
    (w / "quality" / "quality_rf2_noslip.json").unlink()
    r = FR.report(["llg0000"])["chunks"]["llg0000"]
    assert r["state"] == "FAIL" and r["failures"] and "dancers" not in r            # the front's share is gated before the fit's numbers exist
    assert FR.two_view_failures(FR.two_view_share(fr_ll / "nothing")) == ["two-view: no mamma/prepare.json (the front's outputs are not pulled)"]


def test_fit_report_of_the_other_takes_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(FR, "PACKAGE_ROOT", tmp_path)
    monkeypatch.setattr(FR, "DANCE", "carlos_aline")
    monkeypatch.setattr(FR, "TWO_VIEW_MIN_SHARE", None)
    w = chunk_take(tmp_path, fitted_01=0.0)
    w.rename(tmp_path / "work" / "carlos_aline_cag0000")
    rep = FR.report(["cag0000"])
    r = rep["chunks"]["cag0000"]
    assert rep["gates"] is FR.GATES and "two_view" not in r and r["state"] == "pass" and r["failures"] == []
    assert len(FR.table(rep).splitlines()[0].split("|")) == 10


def test_only_the_two_view_take_turns_the_share_gate_on(monkeypatch):
    orig = os.environ.get("HM_FRONT_DANCE")
    session_fr = dict(vars(FR))   # the module as the session imported it
    try:
        for dance, want in (("lucas_luiza", 0.7), ("carlos_aline", None), ("larissa_kadu_recap", None)):
            monkeypatch.setenv("HM_FRONT_DANCE", dance)
            importlib.reload(FR)
            assert FR.TWO_VIEW_MIN_SHARE == want and FR.DANCE == dance
    finally:                                                          # the module as the session imported it
        if orig is None:
            monkeypatch.delenv("HM_FRONT_DANCE", raising=False)
        else:
            monkeypatch.setenv("HM_FRONT_DANCE", orig)
        vars(FR).clear()          # not a reload: that would re-read an HM_FRONT_DANCE another module set at import (see above)
        vars(FR).update(session_fr)


def test_ll_chunks_carry_their_slice_of_a_piecewise_offset(ll_env):
    """Review of 303da6b: a delta(t) taken at the chunk's centre is up to 12 ms off inside a 52 s chunk; each chunk now
    carries the curve over its window and recap_front.fixed_sync_doc writes it as piecewise (mamma_dyn reads it)."""
    from dancecap import mamma_dyn as md
    from dancecap import recap_front as RF
    root, _, spec = ll_env
    curve = [[30.0, -0.066], [80.0, -0.081], [120.0, -0.075], [140.0, -0.061], [200.0, -0.061]]
    write_offsets(root / "work", use_rate=False, piecewise=curve)
    plan = cap.make_plan(spec, hash_inputs=False, code={}, P=cap.LL)
    chunks = [j for j in plan["_jobs"] if j.get("what") == "chunk"]
    for j in chunks:
        sf = j["chunk"]["sync_fixed"]
        t0, t1 = j["chunk"]["k0"] / 30.0, (j["chunk"]["k1"] + 1) / 30.0
        pw = np.asarray(sf["piecewise_ms"]["01"], float)
        assert pw[0, 0] == pytest.approx(t0, abs=1e-3) and pw[-1, 0] == pytest.approx(t1, abs=1e-3) and "02" not in sf["piecewise_ms"]
        cv = np.asarray(curve, float)
        tt = np.linspace(t0, t1, 50)
        assert np.allclose(np.interp(tt, pw[:, 0], pw[:, 1]), 1000 * np.interp(tt, cv[:, 0], cv[:, 1]), atol=0.02)
        doc = RF.fixed_sync_doc(Path("/nonexistent"), sf["offsets_ms"], "test", ["01", "02"], master="02", piecewise_ms=sf["piecewise_ms"])
        off = {"delta": doc["offsets"]["01"]["delta"], "piecewise": doc["offsets"]["01"]["piecewise"]}
        assert np.allclose(md.apply_offset(tt, off) - tt, np.interp(tt, cv[:, 0], cv[:, 1]), atol=2e-5)
        assert "piecewise" not in doc["offsets"]["02"]


def test_ll_shape_window_counts_a_camera_only_where_its_track_is_ok(tmp_path, monkeypatch):
    """lucas_luiza's Steadicam has localisation holes (52 % ok): the shape window must not be chosen where it does not see."""
    monkeypatch.setattr(cap, "WORK_ROOT", tmp_path)
    w = tmp_path / "lucas_luiza"
    (w / "people").mkdir(parents=True)
    (w / "cameras").mkdir(parents=True)
    t = 10.0 + np.arange(3000) / 30.0
    for c in ("01", "02"):
        F = len(t)
        np.savez(w / "people" / f"teachers_{c}.npz", t=t, ok=np.ones(F, bool), conf=np.full((F, 2), 0.9), merged=np.zeros((F, 2), bool),
                 floor=np.stack([np.zeros((F, 2)), np.ones((F, 2))], 1))
        np.savez(w / "cameras" / f"{c}.npz", t=t, ok=(t < 40.0) | (c == "02"))      # 01 sees only the first 30 s
    tt = cap.load_teachers(cap.LL)
    assert tt["01"]["cam_ok"][:900].all() and not tt["01"]["cam_ok"][900:].any() and tt["02"]["cam_ok"].all()
    sw = cap.shape_window(tt, cap.LL, t_min=10.0, t_max=100.0)
    assert sw["t0"] < 15.0 and sw["stats"]["both_cameras_both_dancers"] > 0.9
    assert "cam_ok" not in cap.load_teachers(dataclasses.replace(cap.LL, shape_needs_camera_ok=False))["01"]
