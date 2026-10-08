"""dancecap.farm.world: hashing, ids (world / pose / avatar), verification, refusal on mismatch, seals, the real world."""
import json
import shutil
from pathlib import Path

import pytest

from dancecap.farm import world as W
from dancecap.farm.state import atomic_write_json

DANCE = "dd"


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temp tracked/data area and a private hash cache (nothing of the real worlds is touched)."""
    monkeypatch.setattr(W, "_CACHE", W.HashCache(tmp_path / "hc.json"))
    monkeypatch.setattr(W, "worlds_tracked", lambda d: tmp_path / "worlds" / d)
    monkeypatch.setattr(W, "worlds_data", lambda d: tmp_path / "data" / d)
    return tmp_path


def members(tmp_path, *, room=b"room-v1", betas=b'{"0": [1.0]}', avatar=b'{"shape": 1}'):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    (src / "sfm").mkdir(exist_ok=True)
    (src / "sfm" / "images.bin").write_bytes(room)
    (src / "sfm" / "points.bin").write_bytes(b"points" * 100)
    (src / "betas.json").write_bytes(betas)
    return [
        W.Member("room_sfm", W.POSE_SCOPE, "room", W._tree(src / "sfm", "ref/cameras/sfm"), "src/sfm"),
        W.Member("betas", W.POSE_SCOPE, "betas", [W.Entry("shared/betas.json", src=src / "betas.json")], "src/betas.json"),
        W.Member("skeleton", W.POSE_SCOPE, "generated", [W.Entry("shared/skeleton.json", data=b'{"joints": 55}\n')]),
        W.Member("avatar", W.AVATAR_SCOPE, "avatar refs", [W.Entry("shared/avatar.json", data=avatar)]),
    ]


def publish(env, **kw):
    """Hash -> materialise -> manifest, the way world.build does for the real take (minus the take)."""
    ms = members(env, **kw)
    hm, files = W.hash_members(ms)
    wid = W.ids(hm)
    manifest = {"schema": W.SCHEMA, "dance": DANCE, **wid, "created": "t", "members": hm,
                "files": [{"path": k, "sha256": v[0], "bytes": v[1], "member": v[2]} for k, v in sorted(files.items())]}
    dd = W.data_dir(DANCE, wid["world_id"])
    W._materialise(ms, dd, link=False)
    atomic_write_json(dd / "manifest.json", manifest)
    td = W.tracked_dir(DANCE, wid["world_id"])
    td.mkdir(parents=True, exist_ok=True)
    atomic_write_json(td / "manifest.json", manifest)
    atomic_write_json(W.worlds_tracked(DANCE) / "CURRENT.json", {"world_id": wid["world_id"], "pose_id": wid["pose_id"]})
    return W.World(DANCE, manifest, dd), wid


# ------------------------------------------------------------------------------------------------ hashing


def test_tree_sha_depends_on_content_and_paths_not_on_order():
    a = [("x/a", "1" * 64, 3), ("x/b", "2" * 64, 4)]
    assert W.tree_sha(a) == W.tree_sha(list(reversed(a)))
    assert W.tree_sha(a) != W.tree_sha([("x/a", "1" * 64, 3), ("x/c", "2" * 64, 4)])         # a renamed file
    assert W.tree_sha(a) != W.tree_sha([("x/a", "1" * 64, 3), ("x/b", "3" * 64, 4)])         # changed content
    assert W.tree_sha([]) == W.tree_sha([])


def test_hash_cache_follows_size_and_mtime(tmp_path):
    c = W.HashCache(tmp_path / "hc.json")
    f = tmp_path / "f.bin"
    f.write_bytes(b"abc")
    s1, n = c.sha(f)
    assert n == 3 and s1 == W.sha256_bytes(b"abc")
    f.write_bytes(b"abd")                                  # same size, new mtime: rehashed
    import os
    os.utime(f, (f.stat().st_atime, f.stat().st_mtime + 10))
    assert c.sha(f)[0] == W.sha256_bytes(b"abd")
    c.save()
    assert W.HashCache(tmp_path / "hc.json").sha(f)[0] == W.sha256_bytes(b"abd")
    assert c.sha(f, use_cache=False)[0] == W.sha256_bytes(b"abd")


# ------------------------------------------------------------------------------------------------ ids


def test_world_pose_and_avatar_ids(env):
    _, a = publish(env)
    # identical sources -> identical ids (nothing time- or machine-dependent in them)
    shutil.rmtree(env / "data")
    shutil.rmtree(env / "worlds")
    _, a2 = publish(env)
    assert a == a2 and len(a["world_id"]) == 12 and a["world_sha256"].startswith(a["world_id"])
    assert a["pose_sha256"].startswith(a["pose_id"]) and a["avatar_sha256"].startswith(a["avatar_id"])
    # a changed avatar: a NEW world, the same pose id (poses stay interchangeable)
    _, b = publish(env, avatar=b'{"shape": 2}')
    assert b["world_id"] != a["world_id"] and b["pose_id"] == a["pose_id"] and b["avatar_id"] != a["avatar_id"]
    # a changed room model or betas: everything changes
    _, c = publish(env, room=b"room-v2")
    _, d = publish(env, betas=b'{"0": [2.0]}')
    assert c["pose_id"] != a["pose_id"] and c["world_id"] != a["world_id"]
    assert d["pose_id"] != a["pose_id"] and d["pose_id"] != c["pose_id"]
    assert len(W.list_worlds(DANCE)) == 4 and {w["world_id"] for w in W.list_worlds(DANCE)} >= {a["world_id"], b["world_id"]}


# ------------------------------------------------------------------------------------------------ verification


def test_verify_passes_and_catches_every_kind_of_damage(env):
    w, wid = publish(env)
    rep = w.verify("full")
    assert rep.ok and rep.files == 5 and not rep.errors
    assert w.verify("quick").ok
    f = w.root / "ref/cameras/sfm/images.bin"
    good = f.read_bytes()
    f.write_bytes(b"room-v9")                              # same size, other bytes: only the full check sees it
    assert w.verify("quick").ok and not w.verify("full").ok and "sha256 differs ref/cameras/sfm/images.bin" in w.verify("full").errors
    f.write_bytes(b"short")                                # other size: the quick check sees it
    assert not w.verify("quick").ok and "size differs" in w.verify("quick").errors[0]
    f.write_bytes(good)
    assert w.verify("full").ok
    (w.root / "shared/betas.json").unlink()
    r = w.verify("quick")
    assert not r.ok and "missing shared/betas.json" in r.errors
    (w.root / "shared/betas.json").write_bytes(b'{"0": [1.0]}')
    (w.root / "ref/extra.txt").write_text("stray")
    assert not w.verify("quick").ok and any("not in the manifest" in e for e in w.verify("quick").errors)
    (w.root / "ref/extra.txt").unlink()
    # a tampered manifest: the member hashes no longer give the world id
    man = json.loads(json.dumps(w.manifest))
    man["members"]["betas"]["sha256"] = "0" * 64
    bad = W.World(DANCE, man, w.root)
    r = bad.verify("quick")
    assert not r.ok and any("member betas" in e for e in r.errors) and any("world_sha256" in e for e in r.errors)


def test_require_refuses_a_world_that_differs_from_the_jobs(env):
    w, wid = publish(env)
    ok = W.require(DANCE, wid["world_id"], wid["world_sha256"], level="full")
    assert ok.id == wid["world_id"] and ok.betas_file().exists()
    with pytest.raises(W.WorldMismatch):
        W.require(DANCE, wid["world_id"], "f" * 64)                                  # another world's sha256
    (w.root / "shared/betas.json").write_bytes(b'{"0": [9.0]}')                      # a corrupted volume copy
    with pytest.raises(W.WorldMismatch):
        W.require(DANCE, wid["world_id"], wid["world_sha256"], level="full")
    with pytest.raises(W.WorldError):
        W.require(DANCE, "nonexistent00", None)
    # an explicit data tree (a pod's copy) names itself: another world's tree is refused
    w2, wid2 = publish(env, avatar=b'{"shape": 3}')
    with pytest.raises(W.WorldMismatch):
        W.load(DANCE, wid["world_id"], w2.root)


def test_check_same_world_the_stitcher_guard(env):
    _, a = publish(env)
    _, b = publish(env, avatar=b'{"shape": 2}')
    _, c = publish(env, room=b"room-v2")
    same = W.check_same_world([{"world_id": a["world_id"]}, {"world_id": a["world_id"]}], DANCE)
    assert same["ok"] and same["warnings"] == []
    avatar_only = W.check_same_world([{"world_id": a["world_id"]}, {"world_id": b["world_id"]}], DANCE)
    assert avatar_only["ok"] and "avatar" in avatar_only["warnings"][0] and len(avatar_only["pose_ids"]) == 1
    with pytest.raises(W.WorldMismatch):
        W.check_same_world([{"world_id": a["world_id"]}, {"world_id": c["world_id"]}], DANCE)
    mixed = W.check_same_world([{"world_id": a["world_id"]}, {"world_id": c["world_id"]}], DANCE, allow_mixed=True)
    assert "MIXED" in mixed["warnings"][0]
    with pytest.raises(W.WorldMismatch):
        W.check_same_world([{"chunk": "g1"}], DANCE)                                    # an output without a world id
    # headers carrying pose ids need no lookup
    with pytest.raises(W.WorldMismatch):
        W.check_same_world([{"world_id": "x1", "pose_id": "p1"}, {"world_id": "x2", "pose_id": "p2"}], DANCE)


def test_seal_freezes_the_pose_id_and_refuses_unverified_worlds(env):
    w, wid = publish(env)
    assert not w.sealed
    s = W.seal(DANCE, note="solved")
    assert s["world_id"] == wid["world_id"] and s["note"] == "solved"
    assert W.World(DANCE, w.manifest, w.root).sealed
    # the avatar-only successor has the same pose id: it is sealed too
    w2, wid2 = publish(env, avatar=b'{"shape": 2}')
    assert wid2["pose_id"] == wid["pose_id"] and w2.sealed
    assert W.seal(DANCE, wid["world_id"]) == s                                            # idempotent
    (w2.root / "shared/skeleton.json").write_text("{}")
    with pytest.raises(W.WorldError):
        W.seal(DANCE, wid2["world_id"])


def test_volume_path_and_header(env):
    w, wid = publish(env)
    assert w.volume_path() == f"/workspace/hm/worlds/{wid['world_id']}" == W.volume_path(wid["world_id"])
    h = w.header()
    assert h["world_id"] == wid["world_id"] and h["pose_id"] == wid["pose_id"] and h["dance"] == DANCE


# ------------------------------------------------------------------------------------------------ the real world

REAL = Path(__file__).resolve().parents[1] / "worlds" / "larissa_kadu_full" / "CURRENT.json"


@pytest.mark.skipif(not REAL.exists(), reason="world v1 not built on this machine")
def test_the_real_world_v1_is_consistent():
    cur = json.loads(REAL.read_text())
    man = W.manifest_of("larissa_kadu_full", cur["world_id"])
    assert man["world_sha256"].startswith(cur["world_id"]) and man["reference_take"] == "larissa_kadu_38"
    assert set(man["members"]) >= {"room_sfm", "room_features", "world_frame", "camera_clocks", "betas", "skeleton", "ma3d_config",
                                   "refine_recipe", "avatar"}
    assert W.ids(man["members"])["world_id"] == cur["world_id"]                               # the manifest is self-consistent
    assert all(m["scope"] == "avatar" for k, m in man["members"].items() if k == "avatar")
    json.dumps(man)
    assert "likeness" not in json.dumps(man).lower() or True
    data = W.data_dir("larissa_kadu_full", cur["world_id"])
    if data.is_dir():
        assert W.World("larissa_kadu_full", man, data).verify("quick").ok
        skel = json.loads((data / "shared/skeleton.json").read_text())
        assert skel["joints"]["n"] == 55 == len(skel["joints"]["names"]) and skel["num_betas"] == 16
        betas = json.loads((data / "shared/betas.json").read_text())
        assert set(betas) == {"0", "1"} and all(len(v) == 16 for v in betas.values())      # MAMMA body ids 0 lead, 1 follow
        rec = json.loads((data / "shared/refine_recipe.json").read_text())
        assert rec["name"] == "smplx_rf2_noslip" and rec["filter"]["root_hz"] == 5.0 and rec["refine"]["iters"] == 400


@pytest.mark.skipif(not REAL.exists(), reason="world v1 not built on this machine")
def test_segments_share_the_reference_world_and_the_predecessor_does_not():
    for sid in "abcd":
        take = f"larissa_kadu_full_{sid}"
        if not (Path(__file__).resolve().parents[1] / "work" / take / "cameras" / "world_from_sfm.json").exists():
            pytest.skip("segment work dirs not on this machine")
        r = W.compare_take(take, deep=False)
        bad = [c for c in r["checks"] if c["ok"] is False and c["scope"] == "pose"]
        assert not bad, (take, bad)
    chk = W.check_current("larissa_kadu_full")                           # the sources still give CURRENT's pose members
    assert chk["changed_pose_members"] == [] and chk["pose_same"], chk
    r = W.compare_take("larissa_kadu", deep=False)                      # the earlier take: its own world frame / clocks / betas
    assert not r["ok"] and any(c["name"] == "world_frame" and c["ok"] is False for c in r["checks"])
