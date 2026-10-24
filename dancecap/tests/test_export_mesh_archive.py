"""The mesh-avatar archive of a capture with splat dancers (owner rule 2026-10-23: "I don't ever want to see the basic avatars if gauss
splat avatars are available for a dance. fully archive them when they are replaced with gauss splats."): which files are mesh-only, the
move into work/<take>/archive/mesh_avatars_2026-10-23/<capture>/ and back (archive.json), and export_unity.export never putting them
back into such a capture.

    .venv-nopo4d/Scripts/python.exe -m pytest dancecap/tests/test_export_mesh_archive.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dancecap import export_unity as eu  # noqa: E402
from dancecap.take import PACKAGE_ROOT  # noqa: E402
from _needs import needs_paths  # noqa: E402

KEEP = ("capture.json", "smplx_lead.bin", "smplx_follow.bin", "smplx_lead_skin.bin", "smplx_follow_skin.bin", "hair_groom.json",
        "figure1.json", "figure2.json", "physics.json", "timing.json", "counterbalance.json", "floorcraft.json", "room_tex.png")
MESH = ("lead_albedo.png", "follow_albedo.png", "shoes.json")


def _capture(root: Path, name: str = "07_X", roles=("lead", "follow"), take: str = "t") -> Path:
    cap = root / "SA" / name
    (cap / "splats").mkdir(parents=True)
    for r in roles:
        (cap / "splats" / f"{r}.json").write_text("{}")
        (cap / "splats" / f"{r}.splatav").write_bytes(b"SPAV")
    (cap / "capture.json").write_text(json.dumps({"version": 3, "provenance": {"take": take},
                                                  "smplx_albedo": {"lead": "lead_albedo.png", "follow": "follow_albedo.png"}}))
    for n in KEEP[1:] + MESH:
        (cap / n).write_bytes(f"{n} bytes".encode())
        (cap / f"{n}.meta").write_text(f"guid {n}")
    return cap


def _export_mirror(root: Path, cap: Path) -> Path:
    ex = root / "work" / "export" / cap.name
    ex.mkdir(parents=True)
    for n in ("capture.json", "smplx_lead.bin") + MESH:
        (ex / n).write_bytes(f"export {n}".encode())
    return ex


def test_mesh_only_kinds():
    both = ("lead", "follow")
    assert eu.mesh_only_kind("lead_albedo.png", both) == "albedo"
    assert eu.mesh_only_kind("follow_albedo.png", both) == "albedo"
    assert eu.mesh_only_kind("smplx_follow_face.bin", both) == "face"
    assert eu.mesh_only_kind("shoes.json", both) == "shoes"
    for kept in KEEP + ("audio.wav", "times.json", "room_mesh.bin", "floor_tex.png", "rhythm.json", "zouk-time-analysis.json",
                        "video_meta.json", "lead_albedo.png.meta"):
        assert eu.mesh_only_kind(kept, both) is None, kept
    # one dancer with splats: only its own albedo / face; the shared shoes stay (the other dancer's mesh avatar wears them)
    assert eu.mesh_only_kind("lead_albedo.png", ("lead",)) == "albedo"
    assert eu.mesh_only_kind("follow_albedo.png", ("lead",)) is None
    assert eu.mesh_only_kind("shoes.json", ("lead",)) is None
    assert eu.mesh_only_kind("lead_albedo.png", ()) is None


def test_archive_moves_the_mesh_only_files_and_their_meta(tmp_path):
    cap = _capture(tmp_path)
    ex = _export_mirror(tmp_path, cap)
    adir = tmp_path / "archive"
    before = {n: (cap / n).read_bytes() for n in MESH}
    rep = eu.archive_mesh_avatars(cap, export_dir=ex, take="t", archive_dir=adir)
    assert rep["splat_roles"] == ["lead", "follow"]
    assert sorted(m["name"] for m in rep["moved"]) == sorted(MESH * 2)
    for n in MESH:
        assert not (cap / n).exists() and not (cap / f"{n}.meta").exists()          # nothing mesh-only left, .meta gone too
        assert not (ex / n).exists()
        assert (adir / "StreamingAssets" / n).read_bytes() == before[n]
        assert (adir / "StreamingAssets" / f"{n}.meta").exists()
        assert (adir / "export" / n).read_bytes() == f"export {n}".encode()
    for n in KEEP:
        assert (cap / n).exists() and (cap / "splats" / "lead.splatav").exists(), n
    assert (ex / "capture.json").exists() and (ex / "smplx_lead.bin").exists()
    rec = json.loads((adir / "archive.json").read_text())
    assert rec["state"] == "archived" and rec["rule"] == eu.OWNER_RULE_2026_10_23 and "--restore" in rec["restore"]
    for e in rec["files"]:
        assert set(e) >= {"name", "from", "to", "sha256", "size", "why", "kind"}
        assert len(e["sha256"]) == 64 and e["size"] == eu._abs(e["to"]).stat().st_size
    assert "smplx_<role>_skin.bin" in rec["kept"]
    # idempotent: a second run moves nothing and keeps the record
    again = eu.archive_mesh_avatars(cap, export_dir=ex, take="t", archive_dir=adir)
    assert again["moved"] == []
    assert len(json.loads((adir / "archive.json").read_text())["files"]) == len(rec["files"])


def test_restore_puts_every_file_back(tmp_path):
    cap = _capture(tmp_path)
    ex = _export_mirror(tmp_path, cap)
    adir = tmp_path / "archive"
    snap = sorted((p.relative_to(tmp_path).as_posix(), p.read_bytes()) for p in (tmp_path / "SA").rglob("*") if p.is_file())
    snap_ex = sorted((p.name, p.read_bytes()) for p in ex.iterdir())
    eu.archive_mesh_avatars(cap, export_dir=ex, take="t", archive_dir=adir)
    rep = eu.restore_mesh_avatars(cap, take="t", archive_dir=adir)
    assert len(rep["restored"]) == 6 and rep["left_in_archive"] == []
    assert sorted((p.relative_to(tmp_path).as_posix(), p.read_bytes()) for p in (tmp_path / "SA").rglob("*") if p.is_file()) == snap
    assert sorted((p.name, p.read_bytes()) for p in ex.iterdir()) == snap_ex
    assert json.loads((adir / "archive.json").read_text())["state"] == "restored"


def test_restore_refuses_to_overwrite_a_different_file(tmp_path):
    cap = _capture(tmp_path)
    adir = tmp_path / "archive"
    eu.archive_mesh_avatars(cap, export_dir=tmp_path / "none", take="t", archive_dir=adir)
    (cap / "lead_albedo.png").write_bytes(b"someone else's file")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        eu.restore_mesh_avatars(cap, take="t", archive_dir=adir)
    assert (adir / "StreamingAssets" / "follow_albedo.png").exists()       # nothing moved before the refusal
    assert not (cap / "follow_albedo.png").exists()


def test_one_dancer_with_splats_archives_only_its_own_files(tmp_path):
    cap = _capture(tmp_path, roles=("follow",))
    rep = eu.archive_mesh_avatars(cap, export_dir=tmp_path / "none", take="t", archive_dir=tmp_path / "a")
    assert [m["name"] for m in rep["moved"]] == ["follow_albedo.png"]
    assert (cap / "lead_albedo.png").exists() and (cap / "shoes.json").exists()


def test_a_capture_without_splats_is_refused(tmp_path):
    cap = tmp_path / "SA" / "02_X"
    cap.mkdir(parents=True)
    (cap / "lead_albedo.png").write_bytes(b"x")
    with pytest.raises(ValueError, match="no splats"):
        eu.archive_mesh_avatars(cap, take="t", archive_dir=tmp_path / "a")
    assert (cap / "lead_albedo.png").exists()


def test_a_rewritten_file_supersedes_its_archived_copy(tmp_path):
    cap = _capture(tmp_path)
    adir = tmp_path / "archive"
    eu.archive_mesh_avatars(cap, export_dir=tmp_path / "none", take="t", archive_dir=adir)
    (cap / "shoes.json").write_text("new shoes")                     # e.g. dancecap.shoes export ran again
    rep = eu.archive_mesh_avatars(cap, export_dir=tmp_path / "none", take="t", archive_dir=adir)
    assert [m["name"] for m in rep["moved"]] == ["shoes.json"]
    assert (adir / "StreamingAssets" / "shoes.json").read_text() == "new shoes"
    rec = json.loads((adir / "archive.json").read_text())
    assert len(rec["superseded"]) == 1 and eu._abs(rec["superseded"][0]["kept_as"]).read_bytes() == b"shoes.json bytes"
    assert sum(e["name"] == "shoes.json" for e in rec["files"]) == 1


def test_export_routes_mesh_only_files_to_the_archive_before_the_install():
    """export() archives instead of installing (the guard runs before install_capture and before the report's file list)."""
    import inspect
    src = inspect.getsource(eu.export)
    guard = src.index("archive_mesh_avatars(splat_cap")
    assert guard < src.index("install_capture(out, dst)") and guard < src.index('report["files"]')
    assert "if splat_roles(splat_cap):" in src


# ------------------------------------------------------------------------------- end to end on the synthetic take
SYN = PACKAGE_ROOT / "work" / "synthetic"
needs_synthetic = needs_paths(SYN / "smplx" / "lead.npz", SYN / "smplx" / "follow.npz", SYN / "audio.wav", eu.DEFAULT_MODEL,
                              why="synthetic take (python -m dancecap synthetic) / gated SMPL-X model")


def _synthetic_splat_capture(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    """the synthetic take in a tmp dancecap root (its avatar/ holds fake albedos) and a tmp StreamingAssets capture with splats/ -> (work,
    StreamingAssets, capture)"""
    root = tmp_path / "dancecap"
    work = root / "work" / "synthetic"
    shutil.copytree(SYN / "smplx", work / "smplx")
    for n in ("audio.wav", "truth.json"):
        if (SYN / n).exists():
            shutil.copy2(SYN / n, work / n)
    (work / "avatar").mkdir()
    for r in eu.ROLES:
        (work / "avatar" / f"{r}_albedo.png").write_bytes(f"{r} albedo".encode())
    (root / "takes").mkdir()
    monkeypatch.setattr(eu, "PACKAGE_ROOT", root)
    sa = tmp_path / "SA"
    monkeypatch.setattr(eu, "UNITY_STREAMING", sa)
    cap = sa / "_pytest_mesh_archive"
    (cap / "splats").mkdir(parents=True)
    for r in eu.ROLES:
        (cap / "splats" / f"{r}.json").write_text("{}")
    return work, sa, cap


@needs_synthetic
def test_export_into_a_splat_capture_never_installs_the_albedo(tmp_path, monkeypatch):
    work, sa, cap = _synthetic_splat_capture(tmp_path, monkeypatch)
    root = work.parent.parent
    (cap / "lead_albedo.png").write_bytes(b"installed earlier")           # an older install, not yet archived
    rep = eu.export("synthetic", cap.name, install=True, streaming_assets=sa, room=False, faces=False, shape_edits=False)
    textured = "mesh_avatars_archived" in rep and rep["mesh_avatars_archived"]["files"]
    if not textured:
        pytest.skip("the model has no UVs here: the export wrote no albedo to route")
    assert not any((cap / f"{r}_albedo.png").exists() for r in eu.ROLES)        # never back in the splat capture
    assert not any((Path(rep["out"]) / f"{r}_albedo.png").exists() for r in eu.ROLES)
    assert (cap / "splats" / "lead.json").exists() and (cap / "smplx_lead_skin.bin").exists() and (cap / "capture.json").exists()
    assert "lead_albedo.png" not in rep["files"]
    adir = eu.mesh_archive_dir(cap.name, "synthetic")
    assert adir.is_relative_to(root / "work" / "synthetic" / "archive")
    assert (adir / "export" / "lead_albedo.png").read_bytes() == b"lead albedo"
    assert (adir / "StreamingAssets" / "lead_albedo.png").read_bytes() == b"lead albedo"   # the copy the install would have made
    rec = json.loads((adir / "archive.json").read_text())
    assert len(rec["superseded"]) == 1                                         # the older installed copy is kept beside it
    # the restore gives both folders their albedo back
    eu.restore_mesh_avatars(cap, take="synthetic")
    assert (cap / "lead_albedo.png").read_bytes() == b"lead albedo"
    assert (Path(rep["out"]) / "follow_albedo.png").read_bytes() == b"follow albedo"


@needs_synthetic
def test_validator_counts_archived_files_as_present(tmp_path, monkeypatch):
    """demo_export.validate (behind recap_export / carlos_aline_export / jl_export --validate-only) on an exported + archived synthetic
    splat capture: the albedo and shoes checks pass as 'archived (splat dancers)' instead of failing as missing (fix round 1, M1); with the
    splats taken away a missing albedo fails again."""
    from types import SimpleNamespace
    from dancecap import demo_export as de
    work, sa, cap = _synthetic_splat_capture(tmp_path, monkeypatch)
    (work / "avatar" / "shoes.json").write_text("{}")                 # the take has shoes: without the archive 'shoes' would fail
    (cap / "shoes.json").write_text('{"version": 1}')                 # installed earlier: swept into the archive by the export
    rep = eu.export("synthetic", cap.name, install=True, streaming_assets=sa, room=False, faces=False, shape_edits=False)
    if not (rep.get("mesh_avatars_archived") or {}).get("files"):
        pytest.skip("the model has no UVs here: the export wrote no albedo to route")
    assert not (cap / "shoes.json").exists() and not (cap / "lead_albedo.png").exists()
    assert json.loads((cap / "capture.json").read_text())["smplx_albedo"] == {r: f"{r}_albedo.png" for r in eu.ROLES}
    take = SimpleNamespace(name="synthetic", work=work, cfg={})
    out = Path(rep["out"])
    for folder, names in ((cap, ("lead albedo", "follow albedo", "shoes")), (out, ("lead albedo", "follow albedo"))):
        checks = {c["name"]: c for c in de.validate(folder, take)["checks"]}
        for n in names:
            assert checks[n]["ok"] and "archived (splat dancers)" in str(checks[n]["detail"]), (folder.name, checks[n])
        assert checks["lead motion + skin present"]["ok"]
    shutil.rmtree(cap / "splats")                                      # no splats: the mesh avatars are the dancers again
    checks = {c["name"]: c for c in de.validate(cap, take)["checks"]}
    assert not checks["lead albedo"]["ok"] and not checks["shoes"]["ok"]


def test_archived_from_lists_only_what_left_this_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(eu, "UNITY_STREAMING", tmp_path / "SA")
    cap = _capture(tmp_path)
    ex = _export_mirror(tmp_path, cap)
    adir = tmp_path / "archive"
    assert eu.archived_from(cap, "t", archive_dir=adir) == {}
    eu.archive_mesh_avatars(cap, export_dir=ex, take="t", archive_dir=adir)
    assert sorted(eu.archived_from(cap, "t", archive_dir=adir)) == sorted(MESH)
    mirror = eu.archived_from(ex, "t", archive_dir=adir)               # the mirror has no splats/: its StreamingAssets twin's decide
    assert sorted(mirror) == sorted(MESH) and all("/export/" in e["to"] for e in mirror.values())
    (adir / "StreamingAssets" / "shoes.json").write_text("changed")    # a damaged archived copy does not count
    assert sorted(eu.archived_from(cap, "t", archive_dir=adir)) == sorted(MESH[:2])
    (adir / "StreamingAssets" / "shoes.json").write_bytes(b"shoes.json bytes")
    eu.restore_mesh_avatars(cap, take="t", archive_dir=adir)
    assert eu.archived_from(cap, "t", archive_dir=adir) == {}          # restored: the files are back, nothing archived


def test_a_quest_set_alone_counts_as_splat_data(tmp_path):
    """splat_roles = head-movement CaptureManifest.HasSplatAvatar: splats/<role>.json or any set's splats/<set>/<role>.json"""
    cap = tmp_path / "SA" / "09_X"
    (cap / "splats" / "quest_25k").mkdir(parents=True)
    (cap / "splats" / "quest_25k" / "follow.json").write_text("{}")
    assert eu.splat_roles(cap) == ("follow",)
    (cap / "splats" / "lead.json").write_text("{}")
    assert eu.splat_roles(cap) == ("lead", "follow")


def test_restore_cli_prints_archive_to_original(tmp_path, monkeypatch, capsys):
    cap = _capture(tmp_path)
    adir = tmp_path / "archive"
    monkeypatch.setattr(eu, "mesh_archive_dir", lambda capture, take: adir)
    monkeypatch.setattr(eu, "PACKAGE_ROOT", tmp_path / "pkg")          # no export mirror for this take
    assert eu.main(["archive-avatars", "--capture", str(cap), "--take", "t"]) == 0
    moved = [ln.strip().split("  ->  ") for ln in capsys.readouterr().out.splitlines() if "  ->  " in ln]
    assert len(moved) == 3 and all("/archive/" in b and "/archive/" not in a for a, b in moved)
    assert eu.main(["archive-avatars", "--capture", str(cap), "--take", "t", "--restore"]) == 0
    back = [ln.strip().split("  ->  ") for ln in capsys.readouterr().out.splitlines() if "  ->  " in ln]
    assert len(back) == 3 and all("/archive/" in a and "/archive/" not in b for a, b in back)
    assert (cap / "lead_albedo.png").exists()


@pytest.mark.parametrize("capture,take", [("06_LarissaKaduRecap", "larissa_kadu_recap"), ("07_CarlosAline", "carlos_aline"),
                                          ("08_JohnLula", "john_lula")])
def test_installed_splat_captures_validate(capture, take):
    """the real captures' validator (read-only): their archived albedo / shoes count as present"""
    from dancecap import demo_export as de
    from dancecap.take import Take
    cap = eu.UNITY_STREAMING / capture
    if not (cap / "capture.json").exists() or not eu.splat_roles(cap):
        pytest.skip("capture data not in this checkout (git-ignored likeness data)")
    checks = {c["name"]: c for c in de.validate(cap, Take.load(take))["checks"]}
    for n in ("lead albedo", "follow albedo", "shoes"):
        assert checks[n]["ok"], (n, checks[n])


@pytest.mark.parametrize("capture,take", [("06_LarissaKaduRecap", "larissa_kadu_recap"), ("07_CarlosAline", "carlos_aline"),
                                          ("08_JohnLula", "john_lula")])
def test_installed_splat_captures_are_archived(capture, take):
    cap = eu.UNITY_STREAMING / capture
    if not (cap / "capture.json").exists() or not eu.splat_roles(cap):
        pytest.skip("capture data not in this checkout (git-ignored likeness data)")
    roles = eu.splat_roles(cap)
    left = sorted(p.name for p in cap.iterdir() if p.is_file() and eu.mesh_only_kind(p.name.removesuffix(".meta"), roles))
    assert left == [], f"mesh-only files still in {capture}: {left}"
    rec = json.loads((eu.mesh_archive_dir(capture, take) / "archive.json").read_text())
    assert rec["state"] == "archived"
    for e in rec["files"]:
        assert eu._abs(e["to"]).exists() and eu._sha256(eu._abs(e["to"])) == e["sha256"], e["to"]


# ------------------------------------------------------------------------------- the film library's posters
def _library(tmp_path: Path, film: str = "x") -> tuple[Path, Path, Path]:
    lib = tmp_path / "library"
    (lib / "thumbs").mkdir(parents=True)
    (lib / "library.json").write_text(json.dumps({"version": 1, "films": [{"id": film, "thumbnail": f"thumbs/{film}.jpg"}]}))
    (lib / "thumbs" / f"{film}.jpg").write_bytes(b"mesh poster")
    (lib / "thumbs" / f"{film}.jpg.meta").write_text("guid poster")
    src = tmp_path / "work" / "film" / "review" / "card_thumb.jpg"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"mesh poster")
    presets = tmp_path / "films.toml"
    presets.write_text(f'[{film}]\nid = "{film}"\nthumbnail = "{src.as_posix()}"\n')
    return lib, src, presets


def test_film_poster_paths_names_the_card_and_its_source(tmp_path):
    lib, src, presets = _library(tmp_path)
    assert eu.film_poster_paths("x", library_dir=lib, presets=presets) == [("library", lib / "thumbs" / "x.jpg"), ("source", src)]
    assert eu.film_poster_paths("nope", library_dir=lib, presets=presets) == []


def test_posters_of_the_mesh_avatars_are_archived_and_restored(tmp_path):
    """fix round 1, M2: the library card of a dance with splat dancers shows the splats; the mesh still goes to the archive and back"""
    cap = _capture(tmp_path)
    lib, src, presets = _library(tmp_path)
    targets = eu.film_poster_paths("x", library_dir=lib, presets=presets)
    new = tmp_path / "splat_poster.jpg"
    new.write_bytes(b"splat poster")
    adir = tmp_path / "archive"
    eu.archive_mesh_avatars(cap, export_dir=tmp_path / "none", take="t", archive_dir=adir)
    rep = eu.archive_posters(cap, "x", new, take="t", archive_dir=adir, targets=targets, made="test")
    assert sorted(e["label"] for e in rep["replaced"]) == ["library", "source"]
    assert (lib / "thumbs" / "x.jpg").read_bytes() == b"splat poster" and src.read_bytes() == b"splat poster"
    assert (lib / "thumbs" / "x.jpg.meta").exists()                          # the same Unity asset: its .meta stays
    assert (adir / "posters" / "library" / "x.jpg").read_bytes() == b"mesh poster"
    assert (adir / "posters" / "source" / "card_thumb.jpg").read_bytes() == b"mesh poster"
    rec = json.loads((adir / "archive.json").read_text())
    assert len(rec["posters"]) == 2 and len(rec["files"]) == 3                 # the capture files' record is kept
    assert all(p["replaced_by"]["sha256"] == eu._sha256(new) and p["why"] == eu.POSTER_WHY for p in rec["posters"])
    assert eu.archive_posters(cap, "x", new, take="t", archive_dir=adir, targets=targets)["replaced"] == []   # idempotent
    assert eu.archived_from(cap, "t", archive_dir=adir).keys() == set(MESH)   # posters are not capture files
    back = eu.restore_mesh_avatars(cap, take="t", archive_dir=adir)
    assert len(back["restored"]) == 5
    assert (lib / "thumbs" / "x.jpg").read_bytes() == b"mesh poster" and src.read_bytes() == b"mesh poster"
    assert (adir / "posters" / "library" / "x.splats.jpg").read_bytes() == b"splat poster"   # the splat poster kept for a re-archive
    assert (cap / "lead_albedo.png").exists()


def test_a_poster_replaced_by_something_else_refuses_the_restore(tmp_path):
    cap = _capture(tmp_path)
    lib, src, presets = _library(tmp_path)
    new = tmp_path / "splat_poster.jpg"
    new.write_bytes(b"splat poster")
    adir = tmp_path / "archive"
    eu.archive_posters(cap, "x", new, take="t", archive_dir=adir, targets=eu.film_poster_paths("x", library_dir=lib, presets=presets))
    (lib / "thumbs" / "x.jpg").write_bytes(b"a third poster")
    with pytest.raises(ValueError, match="neither the archived one"):
        eu.restore_mesh_avatars(cap, take="t", archive_dir=adir)
    assert (adir / "posters" / "source" / "card_thumb.jpg").exists() and src.read_bytes() == b"splat poster"   # nothing moved


@pytest.mark.parametrize("film,capture,take", [("recap", "06_LarissaKaduRecap", "larissa_kadu_recap"),
                                               ("carlos_aline", "07_CarlosAline", "carlos_aline"), ("john_lula", "08_JohnLula", "john_lula")])
def test_installed_splat_films_show_splat_posters(film, capture, take):
    """the real library: each poster of a dance with splat dancers (the card and its films.toml source) is the splat still recorded in
    archive.json, and the mesh still is in the archive"""
    cap = eu.UNITY_STREAMING / capture
    adir = eu.mesh_archive_dir(capture, take)
    if not eu.splat_roles(cap) or not (adir / "archive.json").exists():
        pytest.skip("capture data not in this checkout (git-ignored likeness data)")
    rec = json.loads((adir / "archive.json").read_text())
    entries = {e["label"]: e for e in rec.get("posters") or [] if e["film"] == film and not e.get("restored")}
    paths = dict(eu.film_poster_paths(film))
    assert set(entries) == set(paths) == {"library", "source"}
    for label, p in paths.items():
        assert eu._sha256(p) == entries[label]["replaced_by"]["sha256"], p
        assert eu._sha256(eu._abs(entries[label]["to"])) == entries[label]["sha256"]
