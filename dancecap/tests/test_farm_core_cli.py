"""dancecap.farm.cli (core stage): the plug-in mechanism, global options anywhere on the line, status as one JSON line."""
import json
import sys
import types
from pkgutil import ModuleInfo

import pytest

from dancecap.farm import cli


@pytest.fixture(autouse=True)
def clean_plugins():
    cli.PLUGINS.clear()
    hooks = {k: list(v) for k, v in cli.HOOKS.items()}
    yield
    cli.PLUGINS.clear()
    cli.HOOKS.clear()
    cli.HOOKS.update(hooks)


def fake_plugin(monkeypatch, name="fakeplug"):
    mod = types.ModuleType(f"dancecap.farm.{name}")

    def hello(a):
        print(json.dumps({"hello": a.who, "dance": a.dance, "json": a.json}))
        return 0

    def register(sub):
        p = sub.add_parser("hello", help="test plug-in")
        p.add_argument("--who", default="world")
        p.set_defaults(func=hello)
        cli.register_hook("queue_upload", lambda dance, jobs, args: {"x": {"ok": True}})

    mod.register = register
    monkeypatch.setitem(sys.modules, f"dancecap.farm.{name}", mod)
    return mod


def test_a_plugin_registers_a_subcommand_and_hooks_without_editing_the_core(monkeypatch, capsys):
    fake_plugin(monkeypatch)
    monkeypatch.setattr(cli.pkgutil, "iter_modules", lambda path: [ModuleInfo(None, "fakeplug", False), ModuleInfo(None, "state", False),
                                                                   ModuleInfo(None, "nosuchplug", False), ModuleInfo(None, "_private", False)])
    rc = cli.main(["hello", "--who", "farm", "--json", "--dance", "dd"])               # global options may follow the command
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out == {"hello": "farm", "dance": "dd", "json": True}
    assert cli.PLUGINS["fakeplug"]["registered"] is True
    assert "nosuchplug" in cli.PLUGINS and "ModuleNotFoundError" in cli.PLUGINS["nosuchplug"]["error"]    # a broken plug-in is listed, not fatal
    assert "state" not in cli.PLUGINS and "_private" not in cli.PLUGINS                                  # core / private modules are skipped
    assert cli.call_hook("queue_upload", "dd", [], None) == [{"x": {"ok": True}}]


def test_a_failing_hook_never_breaks_the_core_command():
    def boom(*a, **k):
        raise RuntimeError("uploader down")
    cli.register_hook("coverage_updated", boom)
    res = cli.call_hook("coverage_updated", "d", None)
    assert len(res) == 1 and "uploader down" in res[0]["error"]


def test_a_plugin_that_clashes_with_a_core_command_is_reported(monkeypatch):
    mod = types.ModuleType("dancecap.farm.clash")
    mod.register = lambda sub: sub.add_parser("status")                               # argparse: conflicting sub-command
    monkeypatch.setitem(sys.modules, "dancecap.farm.clash", mod)
    monkeypatch.setattr(cli.pkgutil, "iter_modules", lambda path: [ModuleInfo(None, "clash", False)])
    ap = cli.build_parser()
    sub = next(a for a in ap._actions if a.__class__.__name__ == "_SubParsersAction")
    cli.load_plugins(sub)
    assert cli.PLUGINS["clash"]["error"] and not cli.PLUGINS["clash"]["registered"]


def test_status_is_one_json_line_with_the_documented_parts(capsys):
    assert cli.main(["status", "--json"]) == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    s = json.loads(line)
    assert s["dance"] == "larissa_kadu_full" and {"world", "grid", "coverage", "chunks", "pods", "spend", "watcher", "last_events"} <= set(s)
    assert s["grid"]["T"] == 5644 and s["grid"]["cells"] == 189


def test_plan_dry_run_prints_the_plan_and_changes_nothing(capsys, tmp_path):
    before = cli.State("larissa_kadu_full").load()
    assert cli.main(["plan", "--dry-run", "--ignore-s1", "--json"]) in (0, None)
    plan = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert plan["report"]["chunks"] == len(plan["chunks"])
    after = cli.State("larissa_kadu_full").load()
    assert before["chunks"] == after["chunks"] and before["jobs"] == after["jobs"]
