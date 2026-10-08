"""dancecap.pipeline.registry: the stage DAG per capture type, commands from templates only, planned stages."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap.pipeline import registry  # noqa: E402

RUNS_ON = {"local", "local_gpu", "pod", "pod_secure", "human", "unity"}


def test_registry_loads_and_every_dep_resolves():
    sm = registry.stage_map()
    assert len(sm) >= 50
    for s in sm.values():
        assert s["runs_on"] in RUNS_ON, s["id"]
        for d in s["deps"] + s["soft_deps"]:
            assert d in sm, (s["id"], d)


@pytest.mark.parametrize("variant", registry.VARIANTS)
def test_variants_are_acyclic_and_ordered(variant):
    st = registry.variant_stages(variant)
    order = [s["id"] for s in st]
    pos = {sid: i for i, sid in enumerate(order)}
    for s in st:
        for d in s["deps"]:
            assert d in pos and pos[d] < pos[s["id"]], (variant, s["id"], d)
    assert order[0] == "intake"


def test_variant_membership_and_replacements():
    multi = {s["id"] for s in registry.variant_stages("multi")}
    two = {s["id"]: s for s in registry.variant_stages("two")}
    single = {s["id"]: s for s in registry.variant_stages("single")}
    assert {"cam_map", "cam_loc", "sync", "mamma_fit"} <= multi and "pair_calib" not in multi
    assert "pair_calib" in two and "sync" not in two
    assert "pair_calib" in two["mamma_prepare"]["deps"]  # sync's role is taken by pair_calib
    assert not ({"cam_map", "cam_loc", "sync", "mamma_fit"} & set(single))
    assert {"mono_camera", "mono_hmr", "couple_refine"} <= set(single)
    assert "couple_refine" in single["fit_qa"]["deps"]  # mamma_fit -> mono_hmr + couple_refine


def test_planned_stages_are_registered_not_faked():
    sm = registry.stage_map()
    for sid in ("scale_probe", "cam_refine_body", "pair_calib", "mono_camera", "mono_hmr",
                "couple_refine", "classifier"):  # dancer_splats is built (2026-10-08: tests/test_dancer_splats.py)
        s = sm[sid]
        assert s["implemented"] is False and s["runnable"] is None, sid
        assert s.get("planned_reason"), sid
    with pytest.raises(registry.NotRunnable):
        registry.commands("pair_calib", {"take": "x"})
    # the audio clock port is built now (python -m dancecap.pipeline audio-sync)
    assert sm["audio_sync"]["implemented"] and sm["audio_sync"]["runnable"] == "local"
    assert registry.commands("audio_sync", {"take": "x"}) == [["-m", "dancecap.pipeline", "audio-sync", "x"]]


def test_local_commands_come_from_templates():
    cmds = registry.commands("counterbalance", {"take": "abc"})
    assert cmds == [["-m", "dancecap.counterbalance", "detect", "--take", "abc"]]
    for sid, s in registry.stage_map().items():
        if s["runnable"] != "local":
            continue
        ctx = {"take": "abc", "capture": "05_X", "tag": "c8"}
        for argv in registry.commands(sid, ctx):
            assert argv[0] == "-m" and argv[1].startswith("dancecap"), (sid, argv)
            assert not any("{" in a for a in argv), (sid, argv)
    with pytest.raises(registry.NotRunnable):
        registry.commands("demo_export", {"take": "abc"})  # needs a capture name
    with pytest.raises(registry.NotRunnable):
        registry.commands("texture", {"take": "abc"})  # pod stage: only through a session


def test_session_commands_never_run_without_a_cap_and_support_dry_run():
    ctx = {"take": "abc", "max_cost": 4.5, "max_minutes": 200}
    (argv,) = registry.session_command("S2_fit", ["mamma_fit", "texture"], ctx)
    assert argv[:5] == ["-m", "dancecap", "remote", "run", "--take"]
    assert "--max-cost" in argv and argv[argv.index("--max-cost") + 1] == "4.50"
    jobs = argv[argv.index("--jobs") + 1].split(",")
    assert jobs == ["work/abc/mamma_chunks/pod_job.json", "avatar"]
    (dry,) = registry.session_command("S1_front", ["dancers_sam", "cam_loc"], ctx, dry_run=True)
    assert dry[-1] == "--dry-run" and "work/abc/front/pod_job.json" in dry
    assert registry.session_prep("S2_fit", ["texture"], {"take": "abc"}) == [["-m", "dancecap", "texture", "abc",
                                                                             "--write-pod-job"]]


def test_remote_job_names_map_to_stages():
    assert registry.stages_for_remote_job("avatar_fix") == ["texture"]
    assert registry.stages_for_remote_job("mamma") == ["mamma_fit"]
    assert registry.stages_for_remote_job("mamma_loo") == ["mamma_loo"]
    assert "cam_loc" in registry.stages_for_remote_job("front")
    assert registry.stages_for_remote_job("unknown") == []


RETIRED = ("hands_survey", "hands_pod", "hands_build")


def test_hands_refinement_is_retired():
    """The user retired the hands steps (HaMeR / WiLoR / fingers / contact, S3_hands) on 2026-10-07: no stage in any
    variant, no pod session, no remote-job mapping, no dep or soft dep on them, no command that runs a hands module,
    and the export plays the take's own motion (nothing names smplx_detail)."""
    sm = registry.stage_map()
    assert not set(RETIRED) & set(sm)
    for v in registry.VARIANTS:
        assert not set(RETIRED) & {s["id"] for s in registry.variant_stages(v)}, v
    for s in sm.values():
        assert not set(RETIRED) & set(s["deps"] + s["soft_deps"]), s["id"]
        assert s.get("session") != "S3_hands" and s["runs_on"] != "pod_secure", s["id"]
    assert "S3_hands" not in registry.POD_SESSIONS
    assert "S3_hands" not in registry.registry()["pod_sessions"]
    assert not any(x["id"] == "S3_hands" for x in registry.registry()["sessions"])
    assert not any(sp.get("own_runner") or sp.get("runner") for sp in registry.POD_SESSIONS.values())
    for job in ("hands", "hands_pod", "hands_build"):
        assert registry.stages_for_remote_job(job) == [], job
    for sid, ov in registry.OVERLAY.items():
        cmds = json.dumps([ov.get("commands"), ov.get("gate_apply")])
        assert "hands" not in cmds and "hand_" not in cmds, sid
    for sp in registry.POD_SESSIONS.values():
        assert "hands" not in json.dumps(sp), sp
    assert "smplx_detail" not in json.dumps(registry.get("demo_export"))
    (ret,) = [r for r in registry.draft()["retired"] if r.get("session") == "S3_hands"]
    assert ret["ids"] == list(RETIRED) and ret["by"] == "user" and ret["date"] == "2026-10-07"


def test_edges_include_loops_and_helps():
    st = registry.variant_stages("multi")
    e = registry.edges(st)
    assert ["cam_map", "cam_loc", "dep"] in e
    assert any(k.startswith("loop_") for _, _, k in e)
    cam_types = registry.get("camera_types")
    assert any(h["stage"] in ("cam_map", "cam_loc") for h in cam_types["helps"])


def test_resolve_paths_follow_the_conventions(tmp_path):
    from dancecap.pipeline import paths
    paths.set_root(tmp_path / "dc")
    try:
        assert registry.resolve_path("t", "frames/01/frames.json") == tmp_path / "dc" / "work" / "t" / "frames" / "01" / "frames.json"
        assert registry.resolve_path("t", "takes/{take}.toml") == tmp_path / "dc" / "takes" / "t.toml"
        assert registry.resolve_path("t", "prototype/x.json") == tmp_path / "prototype" / "x.json"
        pats = registry.done_patterns(registry.get("demo_export"), "t")
        assert pats[0].endswith("*_demo_export.json")
    finally:
        paths.set_root(None)
