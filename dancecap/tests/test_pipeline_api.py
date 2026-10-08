"""The Pipeline Monitor REST API (dancecap.monitor/1) through the Flask test client: security, takes, the wizard,
videos + camera types, plans and runs, Runpod actions (fake poller, fake delete), review files, triggers.
Never starts a server, never calls runpodctl, never creates a pod."""
import json
import sys
import tomllib
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_fixtures import FakePoller, have_ffmpeg, make_root, make_take, make_video, touch, wait_for  # noqa: E402
from dancecap.pipeline import paths, runs  # noqa: E402
from dancecap.pipeline.monitor import server  # noqa: E402
from dancecap.pipeline.util import redact  # noqa: E402

BASE = "http://127.0.0.1:8790"


class Client:
    def __init__(self, app, token):
        self.c = app.test_client()
        self.token = token
        self.c.get("/", base_url=BASE)  # sets the session cookie

    def get(self, url, **kw):
        return self.c.get(url, base_url=BASE, **kw)

    def send(self, method, url, body=None, headers=None, token=True):
        h = {"X-Monitor-Token": self.token} if token else {}
        h.update(headers or {})
        return getattr(self.c, method)(url, base_url=BASE, json=body, headers=h)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    root = make_root(tmp_path)
    make_take(root)
    started = []

    def fake_start(kind, argv, **kw):
        started.append({"kind": kind, "argv": argv, **kw})
        return {"id": f"r-20261007-120000-{len(started):04x}", "kind": kind, "argv_display": ["python", *argv],
                "state": "starting", "take": kw.get("take"), "stages": kw.get("stages")}

    monkeypatch.setattr(server.runs, "start", fake_start)
    poller = FakePoller()
    m = server.Monitor(port=8790, poller=poller)
    app = server.create_app(m)
    yield {"root": root, "m": m, "app": app, "c": Client(app, m.token), "started": started, "poller": poller}
    m.helper.shutdown(wait=True)
    paths.set_root(None)


def test_security_guards(env):
    c, app = env["c"], env["app"]
    raw = app.test_client()
    assert raw.get("/api/state", base_url="http://evil.example:8790").status_code == 403   # DNS rebinding
    assert raw.get("/api/state", base_url=BASE).status_code == 403                         # no session cookie
    assert raw.get("/api/health", base_url=BASE).status_code == 200                        # health: host check only
    r = c.get("/api/state")
    assert r.status_code == 200 and "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "Access-Control-Allow-Origin" not in r.headers and r.headers["X-Content-Type-Options"] == "nosniff"
    assert c.send("post", "/api/takes/synth/plan", {"target": "ingest"}, token=False).status_code == 403
    assert c.send("post", "/api/takes/synth/plan", {"target": "ingest"},
                  headers={"Origin": "http://evil.example"}).status_code == 403
    r = c.c.post("/api/takes/synth/plan", base_url=BASE, data="target=ingest",
                 headers={"X-Monitor-Token": env["m"].token, "Content-Type": "application/x-www-form-urlencoded"})
    assert r.status_code == 400
    idx = c.get("/")
    assert f'name="dancecap-token" content="{env["m"].token}"' in idx.get_data(as_text=True)
    assert "SameSite=Strict" in idx.headers["Set-Cookie"] and "HttpOnly" in idx.headers["Set-Cookie"]
    assert redact("key rpa_ABCdef123456 and tok AbCdEfGhIjKlMnOpQrStUvWxYz0123456789") == \
        "key rpa_*** and tok AbCd***"
    sha = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
    assert redact(f"bundle {sha}") == f"bundle {sha}"


def test_state_takes_pipeline_and_stage_detail(env):
    c = env["c"]
    st = c.get("/api/state").get_json()
    assert [t["take"] for t in st["takes"]] == ["synth"]
    t = st["takes"][0]
    assert t["variant"] == "multi" and t["n_used"] == 3 and t["camera_types"]["unset"] == 3
    assert any(a["source"] == "camera_types" and a["level"] == "amber" for a in st["alerts"])
    assert st["runpod"]["connection"] == "ok" and st["budget"]["cap"] == 30.0
    tag = c.get("/api/state").headers["ETag"]
    assert c.get("/api/state", headers={"If-None-Match": tag}).status_code == 304
    reg = c.get("/api/stages").get_json()
    assert reg["schema"] == "dancecap.pipeline.stages/1" and "planned" in reg["statuses"]
    snap = c.get("/api/takes/synth/pipeline").get_json()
    s = {x["id"]: x for x in snap["stages"]}
    assert s["camera_types"]["status"] == "blocked" and s["ingest"]["status"] == "ready"
    assert snap["groups"] and snap["critical_path"]
    d = c.get("/api/takes/synth/stages/cam_loc").get_json()
    assert d["def"]["id"] == "cam_loc" and d["camera_type_behaviour"]["tripod"]
    assert d["camera_types"]["handheld_zoom"] == ["01", "02", "03"]
    assert c.get("/api/takes/synth/stages/nope").status_code == 404
    assert c.get("/api/takes/missing").status_code == 404
    detail = c.get("/api/takes/synth").get_json()
    assert detail["toml_etag"].startswith("sha1:") and detail["dancers"]["lead"]["stature_m"] == 1.75
    assert detail["consent"]["lead"]["cloud_processing"] == "?"


def test_take_patch_and_gate_confirm(env):
    c = env["c"]
    etag = c.get("/api/takes/synth").get_json()["toml_etag"]
    r = c.send("patch", "/api/takes/synth", {"dance": {"level": "novice"},
                                              "consent": {"lead": {"cloud_processing": True}}},
               headers={"If-Match": etag})
    assert r.status_code == 200, r.get_json()
    text = (env["root"] / "takes" / "synth.toml").read_text(encoding="utf-8")
    cfg = tomllib.loads(text)
    assert cfg["dance"]["level"] == "novice" and "# level" in text   # the trailing comment is kept
    assert cfg["consent"]["lead"]["cloud_processing"] is True
    assert c.send("patch", "/api/takes/synth", {"dance": {"level": "pro"}},
                  headers={"If-Match": r.get_json()["toml_etag"]}).status_code == 422
    assert c.send("patch", "/api/takes/synth", {"dance": {"level": "novice"}},
                  headers={"If-Match": etag}).status_code == 412
    assert c.send("post", "/api/takes/synth/gates/labels_review", {}).status_code == 200
    ij = json.loads((paths.take_work("synth") / "intake.json").read_text())
    assert ij["labels_reviewed_at"] > 0


def test_camera_type_endpoints(env):
    c = env["c"]
    r = c.get("/api/takes/synth/videos")
    v = r.get_json()["videos"]
    assert r.get_json()["toml_etag"] == r.headers["X-Toml-Etag"]
    assert [x["cam"] for x in v] == ["01", "02", "03"]
    assert v[0]["type"]["state"] == "unset" and v[0]["type"]["effective"]["class"] == "handheld_zoom"
    etag = r.headers["X-Toml-Etag"]
    assert etag == c.get("/api/takes/synth").get_json()["toml_etag"]
    r = c.send("put", "/api/takes/synth/videos/01/camera-type?preview=1", {"mount": "tripod", "zoom": False})
    assert r.status_code == 200 and r.get_json()["invalidated"] == []  # nothing done yet
    assert c.send("put", "/api/takes/synth/videos/01/camera-type", {"mount": "tripod", "zoom": False}).status_code == 400
    r = c.send("put", "/api/takes/synth/videos/01/camera-type", {"mount": "tripod", "zoom": False},
               headers={"If-Match": etag})
    assert r.status_code == 200, r.get_json()
    out = r.get_json()
    assert out["type"]["state"] == "confirmed" and out["type"]["effective"]["class"] == "tripod"
    assert c.send("put", "/api/takes/synth/videos/02/camera-type", {"mount": "handheld", "zoom": True},
                  headers={"If-Match": etag}).status_code == 412  # stale etag
    assert c.send("put", "/api/takes/synth/videos/02/camera-type", {"mount": "gimbal", "zoom": True},
                  headers={"If-Match": out["toml_etag"]}).status_code == 422
    assert c.send("put", "/api/takes/synth/videos/99/camera-type", {"mount": "tripod", "zoom": True},
                  headers={"If-Match": out["toml_etag"]}).status_code == 404
    r = c.get("/api/takes/synth/videos")
    t1 = r.get_json()["videos"][0]["type"]
    assert t1["estimate_delta"]["pod_min_p50"] < 0  # a tripod saves localisation work
    r = c.send("post", "/api/takes/synth/camera-types/accept", {"cams": "all", "min_confidence": 0.7},
               headers={"If-Match": r.headers["X-Toml-Etag"]})
    assert r.status_code == 200
    r = c.send("post", "/api/takes/synth/camera-types/suggest", {"cams": ["02"]})
    assert r.status_code == 202 and env["started"][-1]["argv"] == ["-m", "dancecap.pipeline", "probe", "synth",
                                                                      "--cams", "02"]
    st = c.get("/api/state").get_json()
    assert st["takes"][0]["camera_types"]["confirmed"] == 1
    # excluding a video edits the take TOML's exclude list
    etag = c.get("/api/takes/synth").get_json()["toml_etag"]
    r = c.send("patch", "/api/takes/synth/videos/03", {"excluded": True}, headers={"If-Match": etag})
    assert r.status_code == 200 and r.get_json()["exclude"] == ["03"]
    assert c.get("/api/takes/synth").get_json()["n_used"] == 2


def test_plan_and_runs_flow(env):
    c = env["c"]
    p = c.send("post", "/api/takes/synth/plan", {"target": "ingest"}).get_json()
    assert p["plan_hash"].startswith("sha1:") and p["totals"]["n_run"] == 1 and p["sessions"] == []
    assert c.send("post", "/api/takes/synth/plan", {"target": "bogus stage"}).status_code == 422
    r = c.send("post", "/api/takes/synth/runs", {"plan_hash": p["plan_hash"], "dry_run": True})
    assert r.status_code == 202, r.get_json()
    run = env["started"][-1]
    assert run["kind"] == "dry_run" and run["argv"][:4] == ["-m", "dancecap.pipeline", "run", "synth"]
    assert run["argv"][-1] == "--dry-run" and run["argv"][5].startswith("work/pipeline/plans/")
    r = c.send("post", "/api/takes/synth/runs", {"plan_hash": p["plan_hash"]})
    assert r.status_code == 202 and env["started"][-1]["kind"] == "pipeline"
    touch(paths.take_work("synth") / "frames" / "01" / "frames.json")
    r = c.send("post", "/api/takes/synth/runs", {"plan_hash": p["plan_hash"]})
    assert r.status_code == 412 and r.get_json()["error"]["code"] == "stale_plan"
    assert r.get_json()["error"]["details"]["plan"]["plan_hash"] != p["plan_hash"]
    assert c.send("post", "/api/takes/synth/runs", {"plan_hash": "sha1:" + "0" * 40}).status_code == 422


def test_narrate_export_and_unity_lock(env):
    c = env["c"]
    r = c.send("post", "/api/takes/synth/narrate", {"camera": "02"})
    assert r.status_code == 202
    assert env["started"][-1]["argv"] == ["-m", "dancecap.moves", "annotate", "--take", "synth", "--no-browser",
                                         "--camera", "02"]
    assert c.send("post", "/api/takes/synth/narrate", {"camera": "2; rm"}).status_code == 400
    assert c.send("post", "/api/takes/synth/export", {}).status_code == 422  # no capture yet
    (paths.work() / "unity.lock").mkdir()
    (paths.work() / "unity.lock" / "owner.txt").write_text("another workflow\n")
    r = c.send("post", "/api/takes/synth/export", {"capture": "09_Synth"})
    assert r.status_code == 423 and r.get_json()["error"]["details"]["owner"] == "another workflow"
    r = c.send("post", "/api/takes/synth/export", {"capture": "09_Synth", "smplx_dir": "smplx",
                                                   "override_lock": True})
    assert r.status_code == 202
    assert env["started"][-1]["argv"] == ["-m", "dancecap.pipeline", "export", "synth", "--capture", "09_Synth",
                                         "--smplx-dir", "smplx"]
    # default: the take's own motion (takes/<take>.toml [export] smplx_dir, else smplx/); smplx_detail (the retired
    # hands refinement, 2026-10-07) is refused
    r = c.send("post", "/api/takes/synth/export", {"capture": "09_Synth", "override_lock": True})
    assert r.status_code == 202 and "--smplx-dir" not in env["started"][-1]["argv"]
    assert c.send("post", "/api/takes/synth/export", {"capture": "09_Synth", "smplx_dir": "smplx_detail",
                                                      "override_lock": True}).status_code == 422
    assert c.send("post", "/api/takes/synth/export", {"capture": "../x"}).status_code == 422
    assert c.get("/api/state").get_json()["locks"]["unity"]["held"] is True


def test_review_files_and_analysis_triggers(env):
    c = env["c"]
    w = paths.take_work("synth")
    touch(w / "review" / "demo" / "sheet.jpg", text="jpg")
    touch(w / "review" / "notes.exe", text="x")
    lst = c.get("/api/takes/synth/review").get_json()
    assert [x["name"] for x in lst] == ["demo/sheet.jpg"]
    assert c.get("/api/takes/synth/review/demo/sheet.jpg").status_code == 200
    for bad in ("..%2F..%2F..%2Ftakes%2Fsynth.toml", "../../takes/synth.toml", "notes.exe"):
        assert c.get(f"/api/takes/synth/review/{bad}").status_code == 404
    touch(w / "counterbalance" / "counterbalance.json", text=json.dumps({"version": 1, "intervals": [
        {"kind": "counterbalance", "t0": 12.0, "t1": 14.5, "confidence": 0.82, "pivot": [0.1, 0.0, -0.3]},
        {"kind": "pivot", "t0": 16.0, "t1": 17.0, "confidence": 0.4, "pivot": [0, 0, 0]}]}))
    touch(w / "moves" / "labels.json", text=json.dumps({"version": 1, "placeholder": True, "segments": [
        {"t0": 1, "t1": 2, "move": "giro", "provenance": "placeholder"}]}))
    a = c.get("/api/takes/synth/analysis").get_json()
    kinds = {l["kind"]: l for l in a["lanes"]}
    assert set(kinds) == {"counterbalance", "pivot"}  # placeholder labels never trigger
    cb = next(t for t in a["triggers"] if t["kind"] == "counterbalance")
    assert cb["style"] == "solid" and "counterbalance.axis" in cb["layers"] and cb["params"]["pivot_xz"] == [0.1, -0.3]
    assert next(t for t in a["triggers"] if t["kind"] == "pivot")["style"] == "dashed"  # below the 0.5 threshold
    from dancecap.pipeline import triggers
    p = triggers.write("synth")
    doc = json.loads(p.read_text())
    assert doc["time_base"] == "reference" and len(doc["triggers"]) == 2
    assert c.get("/api/takes/synth/analysis").get_json()["file"]["exists"] is True


def test_runpod_actions_need_confirmation_and_dancecap_pods(env, monkeypatch):
    c, poller = env["c"], env["poller"]
    poller.pods = [{"id": "abcd1234wxyz", "name": "someone-else"}, {"id": "dcap5678qrst", "name": "dancecap-batch"}]
    deleted = []
    from dancecap import remote
    monkeypatch.setattr(remote, "_delete", lambda pid, reason: deleted.append(pid))
    assert c.send("post", "/api/runpod/pods/abcd1234wxyz/delete", {"confirm": "wxyz"}).status_code == 403
    assert c.send("post", "/api/runpod/pods/dcap5678qrst/delete", {"confirm": "nope"}).status_code == 403
    assert c.send("post", "/api/runpod/pods/dcap5678qrst/delete", {"confirm": "qrst"}).status_code == 200
    assert deleted == ["dcap5678qrst"]
    assert c.send("post", "/api/runpod/pods/BAD!/delete", {"confirm": "x"}).status_code == 404
    assert c.send("post", "/api/runpod/cleanup", {"confirm": "yes"}).status_code == 403
    rp = c.get("/api/runpod").get_json()
    assert rp["connection"] == "ok"
    assert c.send("post", "/api/runpod/refresh", {}).status_code == 200
    assert c.send("post", "/api/runpod/refresh", {}).status_code == 429
    assert c.get("/api/sessions").get_json() == []
    assert c.get("/api/sessions/20261007-000000-nope").status_code == 404
    b = c.get("/api/budget").get_json()
    assert b["level"] in ("green", "amber", "red") and b["cap"] == 30.0


def test_runs_log_and_stop_endpoints(env, tmp_path, monkeypatch):
    c = env["c"]
    monkeypatch.undo()  # real runs.start for this test
    (tmp_path / "mods").mkdir()
    (tmp_path / "mods" / "fake_tool.py").write_text("import time\nprint('[annotate] open http://127.0.0.1:8765/', "
                                                    "flush=True)\nprint('rpa_SECRETKEY123', flush=True)\n"
                                                    "for _ in range(1200):\n    time.sleep(0.05)\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "mods"))
    paths.set_root(env["root"])
    r = runs.start("tool", ["-m", "fake_tool"], take="synth", stages=["narrate"])
    assert wait_for(lambda: c.get(f"/api/runs/{r['id']}").get_json().get("url"), 30) == "http://127.0.0.1:8765/"
    log = c.get(f"/api/runs/{r['id']}/log").get_json()
    assert "rpa_***" in " ".join(log["lines"]) and "SECRETKEY" not in " ".join(log["lines"])
    assert c.send("post", f"/api/runs/{r['id']}/stop", {"mode": "after_step"}).status_code == 409
    assert c.send("post", f"/api/runs/{r['id']}/stop", {"mode": "now"}).status_code == 200
    assert wait_for(lambda: c.get(f"/api/runs/{r['id']}").get_json()["state"] == "stopped", 30)
    assert c.get("/api/runs/r-bad").status_code == 404


@pytest.mark.skipif(not have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")
def test_wizard_scan_probe_create_and_stream(env, tmp_path):
    c = env["c"]
    d = tmp_path / "newdance"
    make_video(d / "01-front.mp4", 6.0)
    make_video(d / "02-side.mp4", 6.0)
    (d / "notes.txt").write_text("not media")
    r = c.send("post", "/api/fs/scan", {"folder": str(d)})
    assert r.status_code == 200
    hs = r.get_json()["files"]
    assert [h["name"] for h in hs] == ["01-front.mp4", "02-side.mp4"]
    assert all("handle" in h and h["media"].startswith("/api/media/h-") for h in hs)
    # probe A (the camera-type suggestion) takes a few seconds per clip, minutes when the 4-CPU PC is saturated
    assert wait_for(lambda: all(c.get(f"/api/staging/{h['handle']}").get_json()["probe_state"] == "done" for h in hs), 180)
    m = c.get(f"/api/media/{hs[0]['handle']}", headers={"Range": "bytes=0-99"})
    assert m.status_code == 206 and m.headers["Content-Range"].startswith("bytes 0-99/")
    assert c.send("post", "/api/fs/scan", {"folder": "relative/path"}).status_code == 400
    req = {"take": "new_dance", "videos": [{"handle": hs[0]["handle"], "cam": "01", "type": {"mount": "tripod", "zoom": False}},
                                           {"handle": hs[1]["handle"], "cam": "02"}],
           "dancers": {"lead": {"name": "A", "stature_m": "180 cm"}, "follow": {"name": "B", "stature_m": "5'4\""}},
           "dance": {"level": "novice", "category": "lesson"}, "consent": {}}
    r = c.send("post", "/api/takes", {**req, "dry_run": True})
    assert r.status_code == 200 and r.get_json()["variant"] == "two" and "[cameras.\"01\"]" in r.get_json()["toml_preview"]
    bad = dict(req, dancers={"lead": {"name": "A"}, "follow": {"name": "B", "stature_m": "1.6"}})
    r = c.send("post", "/api/takes", bad)
    assert r.status_code == 422 and r.get_json()["error"]["details"]["problems"][0]["field"] == "dancers.lead.stature_m"
    r = c.send("post", "/api/takes", req)
    assert r.status_code == 201, r.get_json()
    assert env["started"][-1]["argv"] == ["-m", "dancecap.pipeline", "probe", "new_dance"]
    assert c.send("post", "/api/takes", req).status_code == 409
    v = c.get("/api/takes/new_dance/videos").get_json()["videos"]
    assert v[0]["type"]["state"] == "confirmed" and v[0]["probe"]["duration_s"] > 5 and v[0]["playable"]
    s = c.get("/api/takes/new_dance/videos/02/stream", headers={"Range": "bytes=10-"})
    assert s.status_code == 206
    assert c.get("/api/takes/new_dance/videos/07/stream").status_code == 404
    snap = c.get("/api/takes/new_dance/pipeline").get_json()
    st = {x["id"]: x["status"] for x in snap["stages"]}
    assert snap["variant"] == "two" and st["pair_calib"] == "planned" and st["audio_sync"] == "ready"


def test_window_patch(env):
    c = env["c"]
    etag = c.get("/api/takes/synth").get_json()["toml_etag"]
    assert c.send("patch", "/api/takes/synth", {"window": [20, 10]}, headers={"If-Match": etag}).status_code == 422
    r = c.send("patch", "/api/takes/synth", {"window": [11.5, 19.25]}, headers={"If-Match": etag})
    assert r.status_code == 200
    cfg = tomllib.loads((env["root"] / "takes" / "synth.toml").read_text(encoding="utf-8"))
    assert cfg["window"] == [11.5, 19.25] and cfg["name"] == "synth"
