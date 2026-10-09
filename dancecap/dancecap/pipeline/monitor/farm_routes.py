"""Pipeline Monitor, farm routes (docs/FARM.md section 7). Registered from server.create_app (one additive call).

    GET  /api/farm?dance=&demo=        the bundle of one dance: coverage cells, chunks, jobs, queue, pods, budget, volume, world
    GET  /api/farm/log?dance=&name=    tail of a farm log (watch.log by default)
    POST /api/farm/fill/plan           Fill gaps step 1 (no side effects): chunks, pods, price, time, worst case, budget gate
    POST /api/farm/fill/run            Fill gaps step 2: {dance, plan_hash, approve: {max_cost_usd}}
    POST /api/farm/pull                Pull now
    POST /api/farm/watcher/restart     Restart watcher
    POST /api/farm/pods/stop           Stop pods: {dance, pods?, confirm: "STOP"} (farm pods of that dance only)
    POST /api/farm/coverage/refresh    recompute coverage.json now
    POST /api/farm/map-png             {dance, png_b64}: save the map the window drew to farm/review/
    POST /api/farm/reveal              {dance, what: map|logs|farm}: Explorer

Same security model as the rest of the monitor (server.guard: Host allowlist, token + JSON + Origin on writes, cookie on GETs).
The client names an action; the command lines are built in farm_actions.py, never sent by the client.
"""
from __future__ import annotations

import base64
import binascii
import os
import subprocess
import threading
import time

from .. import farm_actions, farm_view, paths
from ..util import TAKE_RE

NO_WINDOW = 0x08000000 if os.name == "nt" else 0
BUNDLE_TTL_S = 1.2        # the bundle is rebuilt at most this often (several windows polling share it)


class _RunpodCache:
    """The server's runpod summary, refreshed in a helper thread so a slow `runpodctl` never stalls the 2 s farm poll."""

    def __init__(self, fetch, ttl: float = 8.0):
        self.fetch, self.ttl = fetch, ttl
        self.value: dict | None = None
        self.at = 0.0
        self._lock = threading.Lock()
        self._busy = False

    def _run(self):
        try:
            v = self.fetch()
            with self._lock:
                self.value, self.at = v, time.time()
        except Exception:  # noqa: BLE001 - keep the last good value
            pass
        finally:
            self._busy = False

    def get(self, wait: float = 4.0) -> dict | None:
        now = time.time()
        if self.value is None or now - self.at > self.ttl:
            if not self._busy:
                self._busy = True
                t = threading.Thread(target=self._run, daemon=True, name="farm-runpod")
                t.start()
                if self.value is None:
                    t.join(wait)
        return self.value


def register(app, m, ok, body, ApiError, runpod_summary) -> None:
    from flask import Response, request

    rpc = _RunpodCache(lambda: runpod_summary())
    farm_actions.set_runpod_provider(lambda: rpc.get(wait=0.0))

    def dance_arg(d: dict | None = None, required: bool = True) -> str:
        v = (d or {}).get("dance") if d is not None else request.args.get("dance")
        v = v or request.args.get("dance")
        if not v:
            v = farm_view.default_dance(show_demo=request.args.get("demo") == "1")
            if not v and required:
                raise ApiError(404, "not_found", "no dance has a farm directory yet")
        if not TAKE_RE.match(v or ""):
            raise ApiError(400, "bad_request", "invalid dance")
        if not (paths.work() / v).is_dir():
            raise ApiError(404, "not_found", f"no dance {v}")
        return v

    def guarded(fn):
        def inner(*a, **k):
            try:
                return fn(*a, **k)
            except farm_actions.ActionError as e:
                raise ApiError(e.status, e.code, e.message, e.details) from None
        inner.__name__ = fn.__name__
        return inner

    # ------------------------------------------------------------------ the bundle
    @app.get("/api/farm")
    def farm_get():
        show_demo = request.args.get("demo") == "1"
        dance = dance_arg()

        def compute():
            rp = rpc.get()
            return farm_view.build(dance, rp=rp, show_demo=show_demo, actions=farm_actions.recent(dance))
        b = m.cached(f"farm:{dance}", BUNDLE_TTL_S, compute)
        out = dict(b)
        dances = farm_view.list_dances(show_demo)
        if not any(d["id"] == dance for d in dances):
            dances.append({"id": dance, "label": dance + (" (demo)" if b.get("demo") else ""), "demo": bool(b.get("demo")),
                           "updated_ts": None, "complete_pct": (b["coverage"].get("percent") or {}).get("complete"),
                           "has_coverage": b["coverage"]["exists"]})
        out["dances"] = dances
        out["logs"] = farm_actions.log_names(dance)
        out["commands"] = {k: farm_actions.have_cmd(k) for k in ("up", "queue", "launch", "watch", "pull", "down")} | {
            "launch": farm_actions.have_cmd("launch") or farm_actions.have_cmd("pods:launch")}
        out["monitor"] = {"instance": m.instance, "port": m.port, "started": m.started}
        try:
            from .server import VERSION, _static_rev
            out["monitor"].update({"version": VERSION, "static_rev": _static_rev()})
        except Exception:  # noqa: BLE001
            pass
        out["rev"] = farm_view.rev_of(out)
        tag = f'"farm-{out["rev"]}"'
        if request.headers.get("If-None-Match") == tag:
            return Response(status=304, headers={"ETag": tag})
        # the map follows what lands: new fit / refine output newer than coverage.json starts a (throttled) recompute
        if b["outputs"]["stale"] and not b.get("demo"):
            try:
                farm_actions.refresh_coverage(dance)
            except farm_actions.ActionError:
                pass
        resp = ok(out)
        resp.headers["ETag"] = tag
        return resp

    @app.get("/api/farm/log")
    @guarded
    def farm_log():
        dance = dance_arg()
        return ok(farm_actions.tail_log(dance, request.args.get("name")))

    # ------------------------------------------------------------------ actions
    @app.post("/api/farm/fill/plan")
    @guarded
    def fill_plan():
        d = body()
        dance = dance_arg(d)
        pods = d.get("pods")
        if pods is not None and (not isinstance(pods, int) or isinstance(pods, bool) or not 1 <= pods <= farm_actions.MAX_PODS):
            raise ApiError(422, "invalid", f"pods must be 1..{farm_actions.MAX_PODS}")
        return ok(farm_actions.fill_plan(dance, pods=pods))

    @app.post("/api/farm/fill/run")
    @guarded
    def fill_run():
        d = body()
        dance = dance_arg(d)
        rec = farm_actions.fill_run(dance, str(d.get("plan_hash") or ""), d.get("approve") or {})
        m.invalidate("farm:")
        return ok(rec, 202)

    @app.post("/api/farm/pull")
    @guarded
    def pull():
        dance = dance_arg(body())
        rec = farm_actions.pull_now(dance)
        m.invalidate("farm:")
        return ok(rec, 202)

    @app.post("/api/farm/watcher/restart")
    @guarded
    def watcher_restart():
        dance = dance_arg(body())
        res = farm_actions.restart_watcher(dance)
        m.invalidate("farm:")
        return ok(res, 202)

    @app.post("/api/farm/pods/stop")
    @guarded
    def pods_stop():
        d = body()
        dance = dance_arg(d)
        ids = d.get("pods")
        if ids is not None and (not isinstance(ids, list) or not all(isinstance(x, str) for x in ids) or len(ids) > 20):
            raise ApiError(422, "invalid", "pods must be a list of pod ids")
        rec = farm_actions.stop_pods(dance, ids or [], str(d.get("confirm") or ""))
        m.invalidate("farm:")
        return ok(rec, 202)

    @app.post("/api/farm/coverage/refresh")
    @guarded
    def coverage_refresh():
        dance = dance_arg(body())
        rec = farm_actions.refresh_coverage(dance, force=True)
        m.invalidate("farm:")
        return ok(rec or {"state": "running", "message": "a coverage refresh is already running"}, 202)

    @app.post("/api/farm/map-png")
    @guarded
    def map_png():
        d = body()
        dance = dance_arg(d)
        try:
            raw = base64.b64decode(str(d.get("png_b64") or ""), validate=True)
        except (binascii.Error, ValueError):
            raise ApiError(400, "bad_request", "png_b64 must be base64") from None
        res = farm_actions.save_map_png(dance, raw)
        return ok({"path": res["path"], "bytes": res["bytes"]})

    @app.post("/api/farm/reveal")
    @guarded
    def reveal():
        d = body()
        dance = dance_arg(d)
        what = d.get("what", "farm")
        fd = farm_view.farm_dir(dance)
        target = {"map": fd / "review", "logs": fd / "logs", "farm": fd}.get(what)
        if what == "map":
            pngs = sorted((fd / "review").glob("monitor_map_*.png")) if (fd / "review").is_dir() else []
            target = pngs[-1] if pngs else fd / "review"
        if target is None or not target.exists():
            raise ApiError(404, "not_found", f"{what} does not exist yet")
        if os.name == "nt":
            args = ["explorer", f"/select,{target}"] if target.is_file() else ["explorer", str(target)]
            subprocess.Popen(args, creationflags=NO_WINDOW)
        return ok({"revealed": paths.rel_display(target)})
