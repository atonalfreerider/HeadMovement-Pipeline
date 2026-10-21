"""Pipeline Monitor backend: Flask app factory, REST API `dancecap.monitor/1` (docs/PIPELINE_MONITOR.md 7).

Served by waitress on 127.0.0.1 only (monitor/__init__.py). Read-only by default: pods, sessions and runs started
elsewhere are only displayed. The only paths to a paid pod are POST /plan (no side effects, returns plan_hash and
p50/p90 cost) followed by POST /runs with {plan_hash, approve: {pods, max_cost_usd}}; the server re-checks the plan,
the budget gate and the consent gate. The only pod deletions are user-confirmed (last 4 characters of the id) on
dancecap pods.

Security (PIPELINE_MONITOR 5): Host allowlist (DNS rebinding), X-Monitor-Token + JSON + Origin on every write, a
SameSite=Strict HttpOnly cookie for GETs and media, no CORS, a strict CSP, commands only from the stage registry
(validated ids; the client never sends an argv), media only for registered takes or picked-file handles, pod JSON
reduced to allowlisted fields, logs redacted. The Runpod key is only ever read inside remote._ctl.

No numpy / torch / cv2 at import time: the first window must paint in ~3 s.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .. import (camtypes, estimate, gates, intake, locks, paths, plan as planmod, registry, runpod_view, runs,
                status, takes, tomledit, triggers)
from ..util import (CAM_RE, CAPTURE_RE, POD_ID_RE, RUN_RE, SID_RE, STAGE_RE, TAKE_RE, read_json, redact, sha1_bytes,
                    write_json_atomic)
from . import media

API = 1
VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
REVIEW_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".json", ".txt"}
MEDIA_EXT = set(takes.VIDEO_EXT) | set(takes.AUDIO_EXT) | set(takes.IMAGE_EXT)
SETTINGS_KEYS = {"shell": ("edge-app", "browser", "pywebview"), "theme": ("dark", "light", "system")}
CSP = ("default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
       "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'")
NO_WINDOW = 0x08000000 if os.name == "nt" else 0


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details or {}


class Monitor:
    """Server state: identity, token, caches, picked-file handles, the pod poller."""

    def __init__(self, port: int = 8790, token: str | None = None, instance: str | None = None, poller=None):
        import uuid
        self.port = port
        self.token = token or secrets.token_urlsafe(32)
        self.cookie = secrets.token_urlsafe(24)
        self.instance = instance or str(uuid.uuid4())
        self.started = time.time()
        self.last_request = time.time()
        self.poller = poller or runpod_view.PodPoller()
        self.handles: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.helper = ThreadPoolExecutor(max_workers=1, thread_name_prefix="monitor-helper")  # UI helper slot
        self.quit = threading.Event()
        self.restart = threading.Event()
        self._cache: dict[str, tuple[float, object]] = {}
        self._gpus: tuple[float, object] | None = None
        self._last_refresh = 0.0
        self.settings = read_json(paths.monitor_dir() / "settings.json", {}) or {}
        self._load_handles()

    # ------------------------------------------------------------------ caches
    def cached(self, key: str, ttl: float, fn):
        now = time.time()
        with self.lock:
            c = self._cache.get(key)
            if c and now - c[0] < ttl:
                return c[1]
        val = fn()
        with self.lock:
            self._cache[key] = (now, val)
        return val

    def invalidate(self, prefix: str = "") -> None:
        with self.lock:
            for k in [k for k in self._cache if k.startswith(prefix)]:
                self._cache.pop(k, None)

    # ------------------------------------------------------------------ picked files
    def _staging_dir(self) -> Path:
        return paths.monitor_dir() / "staging"

    def _load_handles(self) -> None:
        d = self._staging_dir()
        if not d.exists():
            return
        for p in d.glob("h-*.json"):
            h = read_json(p, None)
            if h and h.get("handle") and Path(h.get("path", "")).is_file():
                self.handles[h["handle"]] = h

    def register(self, path: Path, kind: str | None = None) -> dict:
        path = Path(path).resolve()
        ext = path.suffix.lower()
        kind = kind or ("video" if ext in takes.VIDEO_EXT else "audio" if ext in takes.AUDIO_EXT else
                        "image" if ext in takes.IMAGE_EXT else None)
        if kind is None or not path.is_file():
            raise ApiError(400, "bad_request", f"not a media file: {path.name}")
        with self.lock:
            for h in self.handles.values():
                if h["path"] == str(path):
                    return h
            hid = "h-" + secrets.token_hex(8)
            h = {"handle": hid, "kind": kind, "name": path.name, "path": str(path), "size_bytes": path.stat().st_size,
                 "probe": None, "suggestion": None, "probe_state": "queued" if kind != "image" else "done"}
            self.handles[hid] = h
        write_json_atomic(self._staging_dir() / f"{hid}.json", h)
        return h

    def handle(self, hid: str) -> dict:
        if not re.fullmatch(r"h-[0-9a-f]{16}", hid or ""):
            raise ApiError(404, "not_found", "unknown handle")
        h = self.handles.get(hid)
        if not h:
            raise ApiError(404, "not_found", "unknown handle")
        return h

    def probe_handle(self, hid: str) -> None:
        from .. import probe as probemod
        h = self.handles.get(hid)
        if not h or h["kind"] == "image":
            return
        h["probe_state"] = "running"
        try:
            h["probe"] = probemod.probe_file(Path(h["path"]))
            if h["kind"] == "video":
                h["suggestion"] = camtypes.suggest_for_video(Path(h["path"]), h["probe"])
            h["probe_state"] = "done"
        except Exception as e:  # noqa: BLE001
            h["probe_state"], h["error"] = "failed", redact(str(e))[:200]
        write_json_atomic(self._staging_dir() / f"{hid}.json", h)

    def file_handle_out(self, h: dict) -> dict:
        out = {k: h.get(k) for k in ("handle", "kind", "name", "size_bytes", "probe", "suggestion", "probe_state")}
        out["path"] = h["path"].replace("\\", "/")  # display only: the picker returned it to this user
        out["media"] = f"/api/media/{h['handle']}"
        out["strip"] = f"/api/media/{h['handle']}/strip?n=8" if h["kind"] == "video" else None
        if h.get("error"):
            out["error"] = h["error"]
        return out


# ---------------------------------------------------------------------------------------------- helpers

def _clean(o):
    """JSON-safe: NaN / inf -> null, Paths -> workspace-relative display strings."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, Path):
        return paths.rel_display(o)
    return o


def _static_rev() -> str:
    h = hashlib.sha1()
    if STATIC.exists():
        for p in sorted(STATIC.rglob("*")):
            if p.is_file():
                st = p.stat()
                h.update(f"{p.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:8]


def _locks() -> dict:
    gl = paths.work() / "gpu.lock"
    ul = paths.work() / "unity.lock"
    g_owner = None
    if gl.exists():
        try:
            g_owner = gl.read_text(encoding="utf-8", errors="replace").strip()[:120]
        except OSError:
            g_owner = "?"
    u_owner = None
    if ul.exists():
        try:
            u_owner = (ul / "owner.txt").read_text(encoding="utf-8", errors="replace").splitlines()[0][:160] \
                if (ul / "owner.txt").exists() else "?"
        except (OSError, IndexError):
            u_owner = "?"
    return {"gpu": {"held": gl.exists(), "owner": g_owner}, "unity": {"held": ul.exists(), "owner": u_owner}}


def _int_arg(args, name: str, default: int, lo: int, hi: int) -> int:
    """A query integer, clamped; 400 when it is not an integer."""
    v = args.get(name)
    if v is None or v == "":
        return default
    try:
        return max(lo, min(int(v), hi))
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", f"{name} must be an integer") from None


def _float_arg(args, name: str, default: float, lo: float, hi: float) -> float:
    v = args.get(name)
    if v is None or v == "":
        return default
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ApiError(400, "bad_request", f"{name} must be a number") from None
    if not math.isfinite(x):
        raise ApiError(400, "bad_request", f"{name} must be a finite number")
    return max(lo, min(x, hi))


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _take_or_404(take: str) -> None:
    if not TAKE_RE.match(take or "") or not takes.exists(take):
        raise ApiError(404, "not_found", f"no take {take}")


def _unity_info() -> dict:
    proj = paths.unity_project()
    ver = None
    pv = proj / "ProjectSettings" / "ProjectVersion.txt"
    if pv.exists():
        m = re.search(r"m_EditorVersion:\s*(\S+)", pv.read_text(encoding="utf-8", errors="replace"))
        ver = m.group(1) if m else None
    root = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Unity" / "Hub" / "Editor"
    editor = root / ver / "Editor" / "Unity.exe" if ver else None
    running = False
    lf = proj / "Temp" / "UnityLockfile"
    if lf.exists():
        try:
            with open(lf, "a"):
                pass
        except PermissionError:
            running = True
        except OSError:
            pass
    sa = proj / "Assets" / "StreamingAssets"
    caps = sorted(p.name for p in sa.iterdir() if p.is_dir() and (p / "capture.json").exists()) if sa.exists() else []
    return {"project": paths.rel_display(proj), "editor": paths.rel_display(editor) if editor else None,
            "editor_path": editor, "version": ver, "editor_found": bool(editor and editor.exists()),
            "running": running, "lock": _locks()["unity"], "captures": caps}


def _tool_url(run: dict) -> str | None:
    if run.get("url"):
        return run["url"]
    try:
        txt = estimate.read_log_tail(runs.run_dir(run["id"]) / "output.log", 65536)
    except ValueError:
        return None
    m = re.findall(r"\[annotate\] open (http://127\.0\.0\.1:\d+/)", txt)
    return m[-1] if m else None


def _run_out(r: dict) -> dict:
    if r.get("kind") == "tool" and r.get("state") in ("running", "starting"):
        r["url"] = _tool_url(r)
    return r


# ---------------------------------------------------------------------------------------------- app factory

def create_app(m: Monitor):
    from flask import Flask, Response, request

    app = Flask("dancecap-monitor", static_folder=None)
    app.config["JSON_SORT_KEYS"] = False
    app.config["MAX_CONTENT_LENGTH"] = 4 << 20
    app.monitor = m

    def ok(obj, status_code: int = 200, etag: bool = False):
        body = json.dumps(_clean(obj), ensure_ascii=False, default=str, allow_nan=False)
        if etag:
            tag = '"' + hashlib.sha1(body.encode()).hexdigest()[:16] + '"'
            if request.headers.get("If-None-Match") == tag:
                return Response(status=304, headers={"ETag": tag})
            return Response(body, status=status_code, mimetype="application/json", headers={"ETag": tag})
        return Response(body, status=status_code, mimetype="application/json")

    def body() -> dict:
        d = request.get_json(silent=True)
        if d is None:
            if request.data:
                raise ApiError(400, "bad_request", "invalid JSON")
            return {}
        if not isinstance(d, dict):
            raise ApiError(400, "bad_request", "a JSON object is expected")
        return d

    def if_match() -> str:
        v = request.headers.get("If-Match")
        if not v:
            raise ApiError(400, "bad_request", "If-Match (the toml_etag) is required for TOML writes")
        return v.strip('"')

    @app.errorhandler(ApiError)
    def _api_err(e: ApiError):
        return ok({"error": {"code": e.code, "message": e.message, "details": e.details}}, e.status)

    @app.errorhandler(404)
    def _nf(e):
        return ok({"error": {"code": "not_found", "message": "not found", "details": {}}}, 404)

    @app.errorhandler(405)
    def _na(e):
        return ok({"error": {"code": "bad_request", "message": "method not allowed", "details": {}}}, 405)

    @app.errorhandler(Exception)
    def _ie(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):  # 413 (body over 4 MB), 400 from werkzeug, ...: keep their status
            code = e.code or 500
            return ok({"error": {"code": "too_large" if code == 413 else "bad_request" if code < 500 else "internal",
                                 "message": e.description or e.name, "details": {}}}, code)
        import traceback
        traceback.print_exc()
        return ok({"error": {"code": "internal", "message": redact(f"{type(e).__name__}: {e}")[:300], "details": {}}},
                  500)

    # Cookies are scoped per host, not per port: one cookie name per port, so a second server on 127.0.0.1 (a dev
    # server, a monitor on 8791) can never overwrite this one's session cookie.
    cookie_name = f"dcm_{m.port}"

    @app.before_request
    def guard():
        allowed = {f"127.0.0.1:{m.port}", f"localhost:{m.port}", f"[::1]:{m.port}"}
        if request.host not in allowed:
            raise ApiError(403, "forbidden", "host not allowed")
        m.last_request = time.time()
        # details.auth marks "this window's credentials are not this server's" (the UI reloads itself once)
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not secrets.compare_digest(request.headers.get("X-Monitor-Token", ""), m.token):
                raise ApiError(403, "forbidden", "missing or wrong X-Monitor-Token", {"auth": "token"})
            origin = request.headers.get("Origin")
            if origin and origin not in {f"http://{h}" for h in allowed}:
                raise ApiError(403, "forbidden", "cross-origin write")
            if (request.mimetype or request.content_length) and request.mimetype != "application/json":
                raise ApiError(400, "bad_request", "Content-Type must be application/json")
        elif request.path.startswith("/api/") and request.path != "/api/health":
            if not secrets.compare_digest(request.cookies.get(cookie_name, ""), m.cookie):
                raise ApiError(403, "forbidden", "open the monitor page first (session cookie missing)", {"auth": "cookie"})

    @app.after_request
    def headers(resp):
        resp.headers["Content-Security-Policy"] = CSP
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers.pop("Access-Control-Allow-Origin", None)
        if request.path.startswith("/api/") and "Cache-Control" not in resp.headers:
            resp.headers["Cache-Control"] = "no-store"
        if request.method not in ("GET", "HEAD") or resp.status_code >= 400:
            # writes and errors only (the 2 s polls would drown the log); never headers, bodies or query values
            print(f"[req] {time.strftime('%H:%M:%S')} {request.method} {request.path} {resp.status_code}", flush=True)
        return resp

    # ------------------------------------------------------------------ static
    @app.get("/index.html")
    @app.get("/")
    def index():
        p = STATIC / "index.html"
        meta = f'<meta name="dancecap-token" content="{m.token}">'
        if p.exists():
            html = p.read_text(encoding="utf-8")
            html = html.replace("</head>", meta + "\n</head>", 1) if "</head>" in html else meta + html
        else:
            html = (f"<!doctype html><html><head><meta charset='utf-8'><title>Pipeline Monitor</title>{meta}</head>"
                    "<body style='font-family:Segoe UI,sans-serif;background:#111;color:#ddd;padding:24px'>"
                    "<h1>Pipeline Monitor</h1><p>The UI files (monitor/static/) are not installed yet; the API is "
                    "running.</p><ul><li><a style='color:#8cf' href='/api/state'>/api/state</a></li>"
                    "<li><a style='color:#8cf' href='/api/stages'>/api/stages</a></li>"
                    "<li><a style='color:#8cf' href='/api/runpod'>/api/runpod</a></li></ul></body></html>")
        resp = Response(html, mimetype="text/html")
        resp.headers["Cache-Control"] = "no-store"
        resp.set_cookie(cookie_name, m.cookie, httponly=True, samesite="Strict", path="/")
        return resp

    @app.get("/<path:name>")
    def static_file(name):
        if name.startswith("api/"):
            raise ApiError(404, "not_found", "not found")
        p = (STATIC / name).resolve()
        if not str(p).startswith(str(STATIC.resolve()) + os.sep) or not p.is_file():
            raise ApiError(404, "not_found", "not found")
        ctype = {".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".svg": "image/svg+xml",
                 ".json": "application/json", ".webmanifest": "application/manifest+json", ".png": "image/png",
                 ".ico": "image/x-icon", ".html": "text/html"}.get(p.suffix.lower(), "application/octet-stream")
        st = p.stat()
        tag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if request.headers.get("If-None-Match") == tag:
            return Response(status=304, headers={"ETag": tag})
        return Response(p.read_bytes(), mimetype=ctype, headers={"ETag": tag, "Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ system
    @app.get("/api/health")
    def health():
        return ok({"app": "dancecap-monitor", "api": API, "version": VERSION, "instance": m.instance,
                   "pid": os.getpid(), "started": m.started, "static_rev": _static_rev()})

    @app.get("/api/config")
    def config():
        u = _unity_info()
        key = "present" if runpod_view.key_present() else "missing"  # never reads the key itself
        edge = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft" / "Edge" / \
            "Application" / "msedge.exe"
        return ok({"workspace": paths.workspace().name, "root": paths.rel_display(paths.root()),
                   "python": Path(sys.executable).name, "port": m.port, "shell": m.settings.get("shell", "edge-app"),
                   "edge_found": edge.exists(), "unity": {k: u[k] for k in ("project", "editor", "version",
                                                                             "editor_found")},
                   "narration_port": 8765, "cap": estimate.cap(), "reserve": estimate.reserve(),
                   "runpod_key": key, "progress_level": "L0", "settings": m.settings,
                   "compute_policy": "heavy CV/neural work runs on Runpod; this PC: orchestration, UI, light checks "
                                     "(2 local analyses at a time, 1 local GPU stage, 1 pod session)"})

    @app.put("/api/settings")
    def settings():
        d = body()
        s = dict(m.settings)
        for k, v in d.items():
            if k in SETTINGS_KEYS:
                if v not in SETTINGS_KEYS[k]:
                    raise ApiError(422, "invalid", f"{k} must be one of {SETTINGS_KEYS[k]}")
                s[k] = v
            elif k in ("idle_exit_min", "poll_ms", "reserve_usd"):
                if not _is_num(v) or v < 0 or v > 10000:
                    raise ApiError(422, "invalid", f"{k} out of range")
                s[k] = v
            elif k == "unity_editor_root":
                if not isinstance(v, str) or not Path(v).is_dir():
                    raise ApiError(422, "invalid", "unity_editor_root must be an existing folder")
                s[k] = v
            else:
                raise ApiError(422, "invalid", f"unknown setting {k}")
        write_json_atomic(paths.monitor_dir() / "settings.json", s)
        m.settings = s
        return ok(s)

    @app.post("/api/quit")
    def quit_():
        m.quit.set()
        return ok({"quitting": True, "runs_continue": True})

    @app.post("/api/restart")
    def restart():
        m.restart.set()
        m.quit.set()
        return ok({"restarting": True})

    @app.post("/api/shell/open")
    def shell_open():
        d = body()
        from . import open_window
        if d.get("route") is not None:
            route = str(d["route"])
            if not route.startswith("#/") or len(route) > 200 or re.search(r"[\s\"'<>]", route):
                raise ApiError(400, "bad_request", "route must look like #/...")
            open_window(f"http://127.0.0.1:{m.port}/{route}", m.settings.get("shell", "edge-app"))
            return ok({"opened": route})
        rid = str(d.get("tool_run") or "")
        if not RUN_RE.match(rid):
            raise ApiError(400, "bad_request", "route or tool_run required")
        r = runs.describe(rid)
        if not r or r.get("kind") != "tool":
            raise ApiError(404, "not_found", "no such tool run")
        url = _tool_url(r)
        if not url or not re.fullmatch(r"http://127\.0\.0\.1:\d{2,5}/", url):
            raise ApiError(409, "conflict", "the tool has not printed its address yet")
        open_window(url, m.settings.get("shell", "edge-app"))
        return ok({"opened": url})

    @app.post("/api/shell/reveal")
    def shell_reveal():
        d = body()
        take = str(d.get("take") or "")
        _take_or_404(take)
        what = d.get("what", "work")
        target = {"work": paths.take_work(take), "review": paths.take_work(take) / "review",
                  "export": paths.take_work(take) / "export", "toml": paths.take_toml(take)}.get(what)
        if what == "run_log":
            rid = str(d.get("run") or "")
            if not RUN_RE.match(rid):
                raise ApiError(400, "bad_request", "run id required")
            target = runs.run_dir(rid) / "output.log"
        if target is None or not target.exists():
            raise ApiError(404, "not_found", f"{what} does not exist yet")
        if os.name == "nt":
            args = ["explorer", f"/select,{target}"] if target.is_file() else ["explorer", str(target)]
            subprocess.Popen(args, creationflags=NO_WINDOW)
        return ok({"revealed": paths.rel_display(target)})

    # ------------------------------------------------------------------ dashboard
    def take_bundle(take: str):
        def compute():
            ctx = status.context(take)
            snap = status.snapshot(take, ctx)
            return ctx, snap, status.summary(take, snap, ctx)
        return m.cached(f"take:{take}", 1.5, compute)

    def alerts(summaries: list[dict], bud: dict, rp: dict) -> list[dict]:
        out = []
        for s in summaries:
            ct = s["camera_types"]
            unconf = ct.get("unset", 0) + ct.get("suggested", 0)
            if ct.get("conflict"):
                out.append({"id": f"camtypes-conflict:{s['take']}", "level": "red", "source": "camera_types",
                            "message": f"{s['take']}: {ct['conflict']} camera type conflict(s)",
                            "link": f"#/take/{s['take']}/videos", "since": None})
            elif unconf:
                cam_done = dict(s["strip"]).get("cam_loc") in ("done", "stale") or \
                    dict(s["strip"]).get("mono_camera") == "done"
                out.append({"id": f"camtypes:{s['take']}", "level": "info" if cam_done else "amber",
                            "source": "camera_types",
                            "message": f"{s['take']}: {unconf} camera type(s) not confirmed"
                                       + (" (cameras already solved with the default model)" if cam_done else
                                          " - the camera stages use handheld + zoom for them until you confirm"),
                            "link": f"#/take/{s['take']}/videos", "since": None})
            if ct.get("refused"):
                out.append({"id": f"camtypes-refused:{s['take']}", "level": "amber", "source": "camera_types",
                            "message": f"{s['take']}: the cameras stage could not use {ct['refused']} confirmed camera "
                                       "type(s) (the data says the camera moves or zooms): it kept a safer model - "
                                       "check the answer",
                            "link": f"#/take/{s['take']}/videos", "since": None})
            for p in s.get("problems") or []:
                out.append({"id": f"intake:{s['take']}:{p[:20]}", "level": "info", "source": "stage",
                            "message": f"{s['take']}: {p}", "link": f"#/take/{s['take']}", "since": None})
        if bud.get("level") in ("amber", "red"):
            out.append({"id": "budget", "level": bud["level"], "source": "budget",
                        "message": "; ".join(bud.get("messages") or ["budget"]), "link": "#/runpod", "since": None})
        for p in rp.get("pods") or []:
            for f in p.get("flags") or []:
                out.append({"id": f"pod:{p['id']}:{f}", "level": "red", "source": "pod",
                            "message": f"pod {p['id']}: {f}", "link": "#/runpod", "since": None})
        for f in rp.get("sessions_failing") or []:
            out.append({"id": f"session-failing:{f['sid']}:{f['failed']}", "level": "amber", "source": "pod",
                        "message": f"session {f['sid']} ({f.get('take') or '?'}): {f['failed']} pod task(s) failed, "
                                   f"{f['skipped']} skipped after them - its pod bills until the job ends; read the "
                                   "runner log", "link": f"#/runpod/session/{f['sid']}", "since": None})
        if rp.get("connection") in ("api_error",):
            out.append({"id": "runpod", "level": "amber", "source": "pod", "message": rp.get("message") or "Runpod API",
                        "link": "#/runpod", "since": None})
        return out

    def runpod_summary(force: bool = False) -> dict:
        def compute():
            return m.poller.summary(force=force)
        if force:
            m.invalidate("runpod")
        return m.cached("runpod", 5.0, compute)

    @app.get("/api/state")
    def state():
        summaries = []
        for t in takes.list_takes():
            try:
                summaries.append(take_bundle(t)[2])
            except Exception as e:  # noqa: BLE001 - one broken take never hides the others
                summaries.append({"take": t, "error": redact(str(e))[:200]})
        rp = runpod_summary()
        bud = rp.get("budget") or estimate.budget()
        act = [_run_out(r) for r in runs.list_runs(active=True, limit=30)]
        st = {"takes": summaries, "runs_active": act,
              "runpod": {"connection": rp["connection"], "checked": rp.get("checked"), "message": rp.get("message"),
                         "pods_running": rp.get("pods_running", 0), "sessions_active": len(rp.get("sessions_active") or []),
                         "progress_level": rp.get("progress_level")},
              "budget": bud, "alerts": alerts([s for s in summaries if "error" not in s], bud, rp), "locks": _locks(),
              "monitor": {"api": API, "version": VERSION, "static_rev": _static_rev(),
                          "idle_exit_min": m.settings.get("idle_exit_min", 20), "instance": m.instance}}
        st["rev"] = int(hashlib.sha1(json.dumps(st, default=str, sort_keys=True).encode()).hexdigest()[:8], 16)
        st["now"] = time.time()
        resp = ok(st)
        tag = f'"rev-{st["rev"]}"'
        if request.headers.get("If-None-Match") == tag:
            return Response(status=304, headers={"ETag": tag})
        resp.headers["ETag"] = tag
        return resp

    @app.get("/api/stages")
    def stages():
        return ok(registry.public_registry(), etag=True)

    # ------------------------------------------------------------------ takes
    @app.get("/api/takes")
    def takes_list():
        out = []
        for t in takes.list_takes():
            try:
                out.append(take_bundle(t)[2])
            except Exception as e:  # noqa: BLE001
                out.append({"take": t, "error": redact(str(e))[:200]})
        return ok(out, etag=True)

    def _resolve_handles(d: dict) -> dict:
        req = dict(d)
        vids = []
        vin = d.get("videos") or []
        if not isinstance(vin, list):
            raise ApiError(400, "bad_request", "videos: a list of {cam, handle, type?, excluded?}")
        for i, v in enumerate(vin):
            if not isinstance(v, dict):
                raise ApiError(400, "bad_request", f"videos[{i}] must be an object")
            if v.get("type") is not None and not isinstance(v.get("type"), dict):
                raise ApiError(422, "invalid", f"videos[{i}].type must be {{mount, zoom}}",
                               {"problems": [{"field": f"videos[{i}].type", "message": "{mount, zoom}"}]})
            h = m.handle(str(v.get("handle") or ""))
            if h["kind"] != "video":
                raise ApiError(422, "invalid", f"videos[{i}] is not a video",
                               {"problems": [{"field": f"videos[{i}].handle", "message": "not a video"}]})
            vids.append({**{k: v.get(k) for k in ("cam", "excluded", "type")}, "path": h["path"], "probe": h.get("probe"),
                         "suggestion": h.get("suggestion")})
        req["videos"] = vids
        if d.get("soundtrack"):
            if not isinstance(d["soundtrack"], dict):
                raise ApiError(400, "bad_request", "soundtrack: {handle}")
            h = m.handle(str(d["soundtrack"].get("handle") or ""))
            if h["kind"] != "audio":
                raise ApiError(422, "invalid", "the soundtrack must be an audio file (.mp3, .wav, .m4a, ...)",
                               {"problems": [{"field": "soundtrack", "message": "pick an audio file, not a video"}]})
            req["soundtrack"] = {"path": h["path"], "probe": h.get("probe")}
        dancers = {}
        if not isinstance(d.get("dancers") or {}, dict):
            raise ApiError(400, "bad_request", "dancers: {lead: {...}, follow: {...}}")
        for k in ("dance", "consent"):
            if d.get(k) is not None and not isinstance(d.get(k), dict):
                raise ApiError(400, "bad_request", f"{k} must be an object")
        for role, dd in (d.get("dancers") or {}).items():
            if role not in takes.ROLES or not isinstance(dd, dict):
                continue
            dd = dict(dd)
            if isinstance(dd.get("portrait"), dict) and dd["portrait"].get("handle"):
                dd["portrait"] = {"path": m.handle(str(dd["portrait"]["handle"]))["path"]}
            elif dd.get("portrait"):
                raise ApiError(422, "invalid", "portraits are picked files (handles)")
            dancers[role] = dd
        req["dancers"] = dancers
        return req

    @app.post("/api/takes")
    def takes_create():
        d = body()
        dry = bool(d.get("dry_run"))
        req = _resolve_handles(d)
        try:
            res = intake.create(req, dry_run=dry)
        except intake.IntakeError as e:
            if any(p["field"] == "take" and "exists" in p["message"] for p in e.problems):
                raise ApiError(409, "conflict", str(e), {"problems": e.problems})
            raise ApiError(422, "invalid", "the new capture has problems", {"problems": e.problems})
        if dry:
            return ok(res)
        take = res["take"]
        m.invalidate("take:")
        r = runs.start("probe", ["-m", "dancecap.pipeline", "probe", take], take=take, stages=["probe"])
        res["run"] = r
        return ok(res, 201)

    @app.get("/api/takes/<take>")
    def take_detail(take):
        _take_or_404(take)
        ctx, snap, summ = take_bundle(take)
        cfg = ctx["cfg"]
        d = dict(summ)
        dance = cfg.get("dance") or {}
        snd = takes.soundtrack(cfg)
        d.update({"toml_etag": ctx["etag"], "toml_path": f"dancecap/takes/{take}.toml",
                  "dance": {k: dance.get(k) for k in ("level", "category", "song", "artist", "date", "venue")},
                  "dancers": takes.dancers(cfg), "consent": takes.consent(cfg),
                  "soundtrack": {"name": snd["name"], "present": snd["present"]} if snd else None,
                  "window": list(takes.window(cfg)) if takes.window(cfg) else None,
                  "exclude": [str(x) for x in cfg.get("exclude", []) or []], "exports": takes.exports(take),
                  "workload": ctx["wl"]})
        return ok(d, etag=True)

    PATCH_KEYS = {"dance": ("level", "category", "song", "artist", "date", "venue", "song_display"),
                  "dancer": ("name", "stature_m", "shoe_sole_m", "description")}

    def _window_bound(take: str, cfg: dict) -> float | None:
        """End of the latest video on the reference clock (seconds), when the probe and the clocks know it."""
        pj = (read_json(paths.take_work(take) / "probe" / "probe.json", {}) or {}).get("cameras", {})
        ends = []
        for v in takes.videos(take, cfg):
            if v["excluded"]:
                continue
            dur = (pj.get(v["cam"]) or {}).get("duration_s")
            if not _is_num(dur):
                continue
            ck = v.get("clock") or {}
            slope = ck.get("slope") if _is_num(ck.get("slope")) else 1.0
            off = ck.get("offset") if _is_num(ck.get("offset")) else 0.0
            ends.append(slope * float(dur) + off)
        return max(ends) if ends else None

    @app.patch("/api/takes/<take>")
    def take_patch(take):
        _take_or_404(take)
        d = body()
        tag = if_match()
        edits = []
        problems = []
        cfg, _ = takes.load_cfg(take)
        if "dance" in d:
            dd = d["dance"]
            if not isinstance(dd, dict):
                raise ApiError(422, "invalid", "invalid values", {"problems": [{"field": "dance",
                                                                                "message": "an object"}]})
            kv = {k: v for k, v in dd.items() if k in PATCH_KEYS["dance"]}
            if "level" in kv and kv["level"] not in takes.LEVELS:
                problems.append({"field": "dance.level", "message": " | ".join(takes.LEVELS)})
            if "category" in kv and kv["category"] not in takes.CATEGORIES:
                problems.append({"field": "dance.category", "message": " | ".join(takes.CATEGORIES)})
            if any(not isinstance(v, str) or len(v) > 200 for v in kv.values()):
                problems.append({"field": "dance", "message": "text values only (at most 200 characters)"})
            edits.append(("dance", kv))
        ds = d.get("dancers") if "dancers" in d else {}
        if not isinstance(ds, dict):
            raise ApiError(422, "invalid", "invalid values", {"problems": [{"field": "dancers", "message": "an object"}]})
        for role in takes.ROLES:
            dd = ds.get(role)
            if not dd:
                continue
            if not isinstance(dd, dict):
                problems.append({"field": f"dancers.{role}", "message": "an object"})
                continue
            kv = {k: v for k, v in dd.items() if k in PATCH_KEYS["dancer"]}
            if "stature_m" in kv:
                st = intake.parse_stature(kv["stature_m"]) if not isinstance(kv["stature_m"], bool) else None
                if st is None or not 1.2 <= st <= 2.2:
                    problems.append({"field": f"dancers.{role}.stature_m", "message": "a plausible stature"})
                kv["stature_m"] = st
            if "shoe_sole_m" in kv:
                v = kv["shoe_sole_m"]
                try:
                    if isinstance(v, bool):
                        raise ValueError
                    kv["shoe_sole_m"] = float(v)
                    if not (math.isfinite(kv["shoe_sole_m"]) and 0 <= kv["shoe_sole_m"] <= 0.1):
                        raise ValueError
                except (TypeError, ValueError):
                    problems.append({"field": f"dancers.{role}.shoe_sole_m", "message": "0 .. 0.1 m"})
            for k in ("name", "description"):
                if k in kv and (not isinstance(kv[k], str) or len(kv[k]) > 400):
                    problems.append({"field": f"dancers.{role}.{k}", "message": "text"})
            edits.append((f"dancers.{role}", kv))
        if "window" in d:
            w = d["window"]
            if not (isinstance(w, list) and len(w) == 2 and all(_is_num(x) for x in w) and 0 <= w[0] < w[1]):
                problems.append({"field": "window", "message": "[t0, t1] finite reference seconds, 0 <= t0 < t1"})
            else:
                end = _window_bound(take, cfg)
                if end is not None and w[1] > end + 1.0:
                    problems.append({"field": "window", "message": f"t1 is beyond the videos (they end at {end:.1f} s "
                                                                   "on the reference clock)"})
                elif w[1] - w[0] > 6 * 3600:
                    problems.append({"field": "window", "message": "at most 6 hours"})
                else:
                    edits.append((None, {"window": [float(w[0]), float(w[1])]}))
        if "consent" in d:
            c = d["consent"]
            if not isinstance(c, dict):
                raise ApiError(422, "invalid", "invalid values", {"problems": [{"field": "consent",
                                                                                "message": "an object"}]})
            have = takes.consent(cfg)
            kv = {}
            for role in takes.ROLES:
                if role not in c:
                    continue
                if not isinstance(c[role], dict):
                    problems.append({"field": f"consent.{role}", "message": "an object of scopes"})
                    continue
                bad = [k for k in c[role] if k not in takes.CONSENT_SCOPES]
                if bad:
                    problems.append({"field": f"consent.{role}", "message": f"unknown scope(s) {bad}"})
                    continue
                # merge: the scopes not in the request keep their recorded value (never reset to "?")
                merged = {k: have[role][k] for k in takes.CONSENT_SCOPES}
                for k, v in c[role].items():
                    merged[k] = intake._consent_value(v)
                merged["recorded"] = time.strftime("%Y-%m-%d")
                kv[role] = merged
            for k in ("source_rights", "bystanders"):
                if k in c:
                    if not isinstance(c[k], str):
                        problems.append({"field": f"consent.{k}", "message": "text"})
                        continue
                    kv[k] = c[k][:200]
            edits.append(("consent", kv))
        if problems:
            raise ApiError(422, "invalid", "invalid values", {"problems": problems})
        if not edits:
            raise ApiError(400, "bad_request", "nothing to change (dance, dancers, consent, window)")
        try:
            etag = tomledit.write(take, edits, tag)
        except tomledit.StaleEtag as e:
            raise ApiError(412, "stale_etag", "the take TOML changed: reload", {"toml_etag": e.etag})
        m.invalidate(f"take:{take}")
        return ok({"toml_etag": etag})

    @app.get("/api/takes/<take>/toml")
    def take_toml(take):
        _take_or_404(take)
        b = paths.take_toml(take).read_bytes()
        return ok({"etag": sha1_bytes(b), "text": b.decode("utf-8-sig", errors="replace")})

    # ------------------------------------------------------------------ videos + camera types
    def video_probe(take: str, v: dict) -> dict | None:
        pj = (read_json(paths.take_work(take) / "probe" / "probe.json", {}) or {}).get("cameras", {})
        if pj.get(v["cam"]) and not pj[v["cam"]].get("error"):
            return pj[v["cam"]]
        p = v.get("path")
        if not p or not Path(p).exists():
            return None
        cache_p = paths.take_work(take) / "monitor" / "probe_cache.json"
        cache = read_json(cache_p, {}) or {}
        st = Path(p).stat()
        key = f"{st.st_size}|{st.st_mtime_ns}"
        if cache.get(v["cam"], {}).get("key") == key:
            return cache[v["cam"]]["probe"]
        from .. import probe as probemod
        try:
            pr = probemod.probe_file(Path(p))
        except Exception:  # noqa: BLE001
            return None
        cache[v["cam"]] = {"key": key, "probe": pr}
        write_json_atomic(cache_p, cache)
        return pr

    def video_list(take: str) -> list[dict]:
        ctx, snap, _ = take_bundle(take)
        types = {t["cam"]: t for t in ctx["types"]}
        wl = ctx["wl"]
        out = []
        loc_def = registry.get("cam_loc") or {}
        per_cm = float(((loc_def.get("est") or {}).get("wall_min") or {}).get("per_CM", 22.0))
        for v in ctx["videos"]:
            pr = video_probe(take, v)
            cam = v["cam"]
            t = types.get(cam)
            if t is None:
                t = camtypes.load(take, [cam])[0][0]
            cm = (wl.get("per_cam_CM") or {}).get(cam) or 0.0
            delta = (camtypes.loc_factor(t) - 1.0) * per_cm * cm
            t = dict(t, estimate_delta={"pod_min_p50": round(delta, 1),
                                        "usd_p50": round(delta / 60 * estimate.DEFAULT_PRICE["COMMUNITY"], 2),
                                        "basis": "cam_loc work vs handheld + zoom (unit count; no tripod measured yet)"})
            proxy = media.proxy_path(take, cam).exists()
            out.append({"cam": cam, "file": v.get("file"), "excluded": v["excluded"],
                        "exclude_reason": v.get("exclude_reason"), "present": bool(v.get("path") and Path(v["path"]).exists()),
                        "probe": pr, "playable": bool(pr and (pr.get("playable") or proxy)), "proxy": proxy,
                        "stream": f"/api/takes/{take}/videos/{cam}/stream",
                        "strip": f"/api/takes/{take}/videos/{cam}/strip?n=12",
                        "thumb": f"/api/takes/{take}/videos/{cam}/thumb?t=1&w=320",
                        "clock": v.get("clock"), "type": t})
        return out

    @app.get("/api/takes/<take>/videos")
    def videos(take):
        _take_or_404(take)
        ctx, _, _ = take_bundle(take)
        # {videos: [Video], toml_etag} (the etag for If-Match also rides in the X-Toml-Etag header)
        resp = ok({"videos": video_list(take), "toml_etag": ctx["etag"]}, etag=True)
        resp.headers["X-Toml-Etag"] = ctx["etag"]
        return resp

    def video_or_404(take: str, cam: str) -> dict:
        _take_or_404(take)
        if not CAM_RE.match(cam or ""):
            raise ApiError(404, "not_found", "camera ids are two digits")
        cfg, _ = takes.load_cfg(take)
        v = next((x for x in takes.videos(take, cfg) if x["cam"] == cam), None)
        if not v:
            raise ApiError(404, "not_found", f"no camera {cam} in {take}")
        return v

    @app.patch("/api/takes/<take>/videos/<cam>")
    def video_patch(take, cam):
        video_or_404(take, cam)
        d = body()
        tag = if_match()
        if not isinstance(d.get("excluded"), bool):
            raise ApiError(400, "bad_request", "{excluded: true|false}")
        cfg, _ = takes.load_cfg(take)
        ex = [str(x) for x in cfg.get("exclude", []) or []]
        ex = sorted(set(ex) | {cam}) if d["excluded"] else [x for x in ex if x != cam]
        try:
            etag = tomledit.write(take, [(None, {"exclude": ex})], tag)
        except tomledit.StaleEtag as e:
            raise ApiError(412, "stale_etag", "the take TOML changed: reload", {"toml_etag": e.etag})
        m.invalidate(f"take:{take}")
        return ok({"toml_etag": etag, "exclude": ex})

    @app.put("/api/takes/<take>/videos/<cam>/camera-type")
    def camera_type(take, cam):
        video_or_404(take, cam)
        d = body()
        mount, zoom = d.get("mount"), d.get("zoom")
        if mount not in camtypes.MOUNTS or not isinstance(zoom, bool):
            raise ApiError(422, "invalid", "mount must be handheld|tripod and zoom true|false",
                           {"problems": [{"field": "mount", "message": "handheld | tripod"},
                                         {"field": "zoom", "message": "true | false"}]})
        ctx, snap, _ = take_bundle(take)
        stat = {x["id"]: x["status"] for x in snap["stages"]}
        vst = registry.variant_stages(ctx["variant"])
        inval = camtypes.invalidated_by_change(take, cam, mount, zoom, stat,
                                               lambda roots: registry.descendants(vst, roots))
        est50 = sum((next(x for x in snap["stages"] if x["id"] == s)["estimate"]["p50_s"]) for s in inval)
        cost50 = sum((next(x for x in snap["stages"] if x["id"] == s)["estimate"]["cost_p50_usd"]) for s in inval)
        rerun = {"p50_s": round(est50), "pod_usd_p50": round(cost50, 2)}
        if request.args.get("preview") in ("1", "true"):
            return ok({"invalidated": inval, "rerun": rerun, "toml_etag": ctx["etag"]})
        tag = if_match()
        try:
            t, etag = camtypes.set_type(take, cam, mount, zoom, tag)
        except (camtypes.StaleEtag, tomledit.StaleEtag) as e:
            raise ApiError(412, "stale_etag", "the take TOML changed: reload", {"toml_etag": e.etag})
        except camtypes.CameraTypeError as e:
            raise ApiError(422, "invalid", str(e))
        m.invalidate(f"take:{take}")
        return ok({"type": t, "toml_etag": etag, "invalidated": inval, "rerun": rerun})

    @app.post("/api/takes/<take>/camera-types/accept")
    def camera_types_accept(take):
        _take_or_404(take)
        d = body()
        tag = if_match()
        cams = d.get("cams", "all")
        cfg, _ = takes.load_cfg(take)
        used = takes.used_cams(take, cfg)
        if cams == "all":
            cams = used
        elif not isinstance(cams, list) or not all(isinstance(c, str) and CAM_RE.match(c) for c in cams):
            raise ApiError(400, "bad_request", "cams: list of two-digit ids or \"all\"")
        mc = d.get("min_confidence", camtypes.MIN_CONFIDENCE)
        if not _is_num(mc) or not 0.0 <= mc <= 1.0:
            raise ApiError(422, "invalid", "min_confidence: a number in [0, 1]")
        mc = float(mc)
        try:
            types, etag = camtypes.accept(take, cams, mc, tag, all_cams=used)
        except camtypes.StaleEtag as e:
            raise ApiError(412, "stale_etag", "the take TOML changed: reload", {"toml_etag": e.etag})
        m.invalidate(f"take:{take}")
        return ok({"types": types, "toml_etag": etag})

    @app.post("/api/takes/<take>/camera-types/suggest")
    def camera_types_suggest(take):
        _take_or_404(take)
        d = body()
        cams = d.get("cams")
        argv = ["-m", "dancecap.pipeline", "probe", take]
        if cams:
            if not isinstance(cams, list) or not all(isinstance(c, str) and CAM_RE.match(c) for c in cams):
                raise ApiError(400, "bad_request", "cams: list of two-digit ids")
            argv += ["--cams", ",".join(cams)]
        r = runs.start("probe", argv, take=take, stages=["probe"])
        return ok({"run": r}, 202)

    @app.route("/api/takes/<take>/videos/<cam>/stream", methods=["GET", "HEAD"])
    def video_stream(take, cam):
        v = video_or_404(take, cam)
        px = media.proxy_path(take, cam)
        p = px if px.exists() else Path(v["path"]) if v.get("path") else None
        if not p or not p.exists():
            raise ApiError(404, "not_found", "video file not found")
        return media.file_response(p, request.headers.get("Range"), head=request.method == "HEAD")

    @app.get("/api/takes/<take>/videos/<cam>/thumb")
    def video_thumb(take, cam):
        v = video_or_404(take, cam)
        if not v.get("path") or not Path(v["path"]).exists():
            raise ApiError(404, "not_found", "video file not found")
        try:
            p = media.thumbnail(Path(v["path"]), _float_arg(request.args, "t", 1.0, 0.0, 86400.0),
                                _int_arg(request.args, "w", 320, 32, 1920))
        except (ValueError, RuntimeError) as e:
            raise ApiError(400, "bad_request", str(e))
        return media.file_response(p, None, mimetype="image/jpeg")

    @app.get("/api/takes/<take>/videos/<cam>/strip")
    def video_strip(take, cam):
        v = video_or_404(take, cam)
        if not v.get("path") or not Path(v["path"]).exists():
            raise ApiError(404, "not_found", "video file not found")
        pr = video_probe(take, v) or {}
        try:
            p = media.strip(Path(v["path"]), pr.get("duration_s"), _int_arg(request.args, "n", 12, 1, 48),
                            _int_arg(request.args, "h", 90, 32, 480))
        except (ValueError, RuntimeError) as e:
            raise ApiError(400, "bad_request", str(e))
        return media.file_response(p, None, mimetype="image/jpeg")

    @app.post("/api/takes/<take>/videos/<cam>/proxy")
    def video_proxy(take, cam):
        video_or_404(take, cam)
        r = runs.start("media", ["-m", "dancecap.pipeline.monitor.media", "proxy", take, cam], take=take,
                       stages=["proxy"])
        return ok({"run": r}, 202)

    # ------------------------------------------------------------------ picker + staging
    @app.post("/api/fs/pick")
    def fs_pick():
        d = body()
        kind = d.get("kind")
        if kind not in ("videos", "soundtrack", "portrait", "folder"):
            raise ApiError(400, "bad_request", "kind: videos | soundtrack | portrait | folder")
        p = subprocess.run([runs.python_exe(), "-m", "dancecap.pipeline.monitor.pick", "--kind", kind],
                           capture_output=True, text=True, timeout=900, creationflags=NO_WINDOW,
                           stdin=subprocess.DEVNULL, cwd=str(paths.REAL_ROOT))
        try:
            files = json.loads((p.stdout or "").strip().splitlines()[-1]).get("files", [])
        except (IndexError, ValueError):
            files = []
        if kind == "folder":
            return ok({"folder": files[0] if files else None, "files": scan(files[0])["files"] if files else []})
        out = []
        for f in files:
            try:
                h = m.register(Path(f))
                out.append(m.file_handle_out(h))
                if h["probe_state"] == "queued":
                    m.helper.submit(m.probe_handle, h["handle"])
            except ApiError:
                continue
        return ok({"files": out})

    def scan(folder: str) -> dict:
        p = Path(str(folder or "")).expanduser()
        if str(p).startswith("\\\\") or not p.is_absolute() or not p.is_dir():
            raise ApiError(400, "bad_request", "an existing local folder path is required")
        out = []
        for f in sorted(p.iterdir()):
            if f.is_file() and f.suffix.lower() in MEDIA_EXT:
                h = m.register(f)
                if h["probe_state"] == "queued":
                    m.helper.submit(m.probe_handle, h["handle"])
                out.append(m.file_handle_out(h))
        return {"folder": str(p).replace("\\", "/"), "files": out}

    @app.post("/api/fs/scan")
    def fs_scan():
        return ok(scan(body().get("folder")))

    @app.get("/api/staging/<hid>")
    def staging_get(hid):
        return ok(m.file_handle_out(m.handle(hid)))

    @app.post("/api/staging/probe")
    def staging_probe():
        hs = body().get("handles") or []
        if not isinstance(hs, list):
            raise ApiError(400, "bad_request", "handles: [..]")
        for hid in hs:
            h = m.handle(str(hid))
            h["probe_state"] = "queued"
            m.helper.submit(m.probe_handle, h["handle"])
        return ok({"queued": len(hs)}, 202)

    @app.route("/api/media/<hid>", methods=["GET", "HEAD"])
    def media_get(hid):
        h = m.handle(hid)
        return media.file_response(Path(h["path"]), request.headers.get("Range"), head=request.method == "HEAD")

    @app.get("/api/media/<hid>/thumb")
    def media_thumb(hid):
        h = m.handle(hid)
        if h["kind"] == "image":
            return media.file_response(Path(h["path"]), None)
        if h["kind"] != "video":
            raise ApiError(400, "bad_request", "no thumbnail for audio")
        try:
            p = media.thumbnail(Path(h["path"]), _float_arg(request.args, "t", 1.0, 0.0, 86400.0),
                                _int_arg(request.args, "w", 320, 32, 1920))
        except (ValueError, RuntimeError) as e:
            raise ApiError(400, "bad_request", str(e))
        return media.file_response(p, None, mimetype="image/jpeg")

    @app.get("/api/media/<hid>/strip")
    def media_strip(hid):
        h = m.handle(hid)
        if h["kind"] != "video":
            raise ApiError(400, "bad_request", "strips are for videos")
        try:
            p = media.strip(Path(h["path"]), (h.get("probe") or {}).get("duration_s"), _int_arg(request.args, "n", 8, 1, 48),
                            _int_arg(request.args, "h", 90, 32, 480))
        except (ValueError, RuntimeError) as e:
            raise ApiError(400, "bad_request", str(e))
        return media.file_response(p, None, mimetype="image/jpeg")

    # ------------------------------------------------------------------ pipeline
    @app.get("/api/takes/<take>/pipeline")
    def pipeline(take):
        _take_or_404(take)
        snap = dict(take_bundle(take)[1])
        if request.args.get("variants") == "all":
            snap["other_variants"] = {v: [s["id"] for s in registry.variant_stages(v)] for v in registry.VARIANTS}
        return ok(snap, etag=True)

    @app.get("/api/takes/<take>/stages/<stage>")
    def stage_detail(take, stage):
        _take_or_404(take)
        if not STAGE_RE.match(stage or ""):
            raise ApiError(404, "not_found", "no such stage")
        ctx, snap, _ = take_bundle(take)
        x = next((s for s in snap["stages"] if s["id"] == stage), None)
        if x is None:
            raise ApiError(404, "not_found", f"{stage} is not a stage of this take's variant")
        d = dict(x)
        d["def"] = registry.get(stage)
        d["outputs_detail"] = x["outputs"]["detail"]
        st = {s["id"]: s["status"] for s in snap["stages"]}
        d["deps_detail"] = [{"id": dd, "status": st.get(dd), "soft": False} for dd in x["deps"]] + \
            [{"id": dd, "status": st.get(dd), "soft": True} for dd in x.get("soft_deps", [])]
        rr = [_run_out(r) for r in runs.list_runs(take=take, limit=200) if stage in (r.get("stages") or [])][:10]
        d["runs"] = rr
        d["log_tail"] = runs.log_chunk(runs.run_dir(rr[0]["id"]) / "output.log", 0, 16384, tail=True)["lines"] if rr else []
        d["progress_steps"] = status.progress_events(take, stage, rr[0]["id"])[1] if rr else []
        d["warnings"] = x["outputs"].get("soft_newer", []) + [f"upstream stale: {u}" for u in x["outputs"].get("upstream_stale", [])]
        if stage in status.CAMERA_STAGES:
            d["camera_type_behaviour"] = (registry.get(stage) or {}).get("camera_types")
        return ok(d)

    @app.post("/api/takes/<take>/gates/<stage>")
    def gate_confirm(take, stage):
        """Close a human gate: labels_review (one sheet looked at), or a review gate the orchestrator opened after
        running fit_qa / geoveto (gates.py). Body {decision}: fit_qa ok | reject; geoveto apply (runs
        `dancers_geoveto --apply` as a run) | keep | reject. reject marks the stage stale (re-run it or its inputs)."""
        _take_or_404(take)
        d = body()
        if stage == "labels_review":
            p = paths.take_work(take) / "intake.json"
            ij = read_json(p, {}) or {}
            ij["labels_reviewed_at"] = time.time()
            write_json_atomic(p, ij)
            m.invalidate(f"take:{take}")
            return ok({"confirmed": stage})
        if stage not in gates.DECISIONS:
            raise ApiError(400, "bad_request", "gates: labels_review, fit_qa, geoveto (camera types: Video Review; "
                                               "narration: the narration tool)")
        decision = d.get("decision", gates.DECISIONS[stage][0] if len(gates.DECISIONS[stage]) == 1 else None)
        allowed = list(gates.DECISIONS[stage]) + ["reject"]
        if decision not in allowed:
            raise ApiError(422, "invalid", f"decision: {' | '.join(allowed)}")
        if not gates.is_open(take, stage):
            raise ApiError(409, "conflict", f"no open review gate at {stage} (it opens when a pipeline run runs it)")
        run = None
        if decision == "apply":
            if any(r.get("kind") in ("pipeline", "dry_run") for r in runs.list_runs(take=take, active=True)) or \
                    locks.holder(locks.take_lock(take)):
                raise ApiError(409, "conflict", "a pipeline run of this take is active: apply after it ends")
            argv = registry.gate_apply_commands(stage, {"take": take})[0]
            run = runs.start("stage", argv, take=take, stages=[stage])
        g = gates.confirm(take, stage, decision, run=(run or {}).get("id"))
        m.invalidate(f"take:{take}")
        return ok({"confirmed": stage, "decision": decision, "gate": g, "run": run})

    @app.post("/api/takes/<take>/stages/<stage>/accept-inputs")
    def accept_inputs(take, stage):
        """Keep a stale stage's outputs (an upstream rewrite that does not matter): it is fresh again until an
        input changes. Never used for camera-type staleness (that needs a re-run)."""
        _take_or_404(take)
        if not STAGE_RE.match(stage or ""):
            raise ApiError(404, "not_found", "no such stage")
        try:
            rec = status.accept_inputs(take, stage)
        except KeyError:
            raise ApiError(404, "not_found", f"{stage} is not a stage of this take")
        m.invalidate(f"take:{take}")
        return ok({"stage": stage, "accepted": rec})

    @app.post("/api/takes/<take>/plan")
    def make_plan(take):
        _take_or_404(take)
        try:
            p = planmod.make_plan(take, body())
        except planmod.PlanError as e:
            raise ApiError(422, "invalid", str(e))
        planmod.save(p)
        return ok(p)

    @app.post("/api/takes/<take>/runs")
    def start_run(take):
        _take_or_404(take)
        d = body()
        h = str(d.get("plan_hash") or "")
        dry = bool(d.get("dry_run"))
        if any(r.get("kind") in ("pipeline", "dry_run") for r in runs.list_runs(take=take, active=True)):
            raise ApiError(409, "conflict", "a pipeline run of this take is active")
        lk = locks.holder(locks.take_lock(take))
        if lk is not None:
            raise ApiError(409, "conflict", f"a pipeline run of this take is active (pid {lk.get('pid')}, "
                                            f"{'run ' + lk['run'] if lk.get('run') else 'started from the CLI'})")
        rid = runs.new_id()
        try:
            p = planmod.revalidate(take, h, d.get("approve"), dry, run_id=rid)
        except planmod.StalePlan as e:
            raise ApiError(412, "stale_plan", "the plan changed: review it again", {"plan": e.fresh})
        except planmod.ApprovalError as e:
            raise ApiError(409, e.code, str(e), e.details)
        except planmod.PlanError as e:
            raise ApiError(422, "invalid", str(e))
        rel = Path("work") / "pipeline" / "plans" / planmod.plan_path(p["plan_hash"]).name
        argv = ["-m", "dancecap.pipeline", "run", take, "--plan", rel.as_posix()] + (["--dry-run"] if dry else [])
        stages_ = [s["stage"] for s in p["steps"] if s["action"] in ("run", "rerun_stale")]
        r = runs.start("dry_run" if dry else "pipeline", argv, take=take, stages=stages_, plan_hash=p["plan_hash"],
                       run_id=rid)
        m.invalidate(f"take:{take}")
        return ok({"run": r, "plan": p}, 202)

    @app.post("/api/takes/<take>/narrate")
    def narrate(take):
        _take_or_404(take)
        d = body()
        cam = d.get("camera")
        if cam is not None and not CAM_RE.match(str(cam)):
            raise ApiError(400, "bad_request", "camera is a two-digit id")
        for r in runs.list_runs(take=take, active=True, kind="tool"):
            return ok({"run": _run_out(r), "reused": True})
        argv = ["-m", "dancecap.moves", "annotate", "--take", take, "--no-browser"] + (["--camera", cam] if cam else [])
        r = runs.start("tool", argv, take=take, stages=["narrate"])
        return ok({"run": r}, 202)

    @app.post("/api/takes/<take>/export-dry-run")
    def export_dry_run(take):
        """The Export dialog's Dry run (its own route: a server without it answers 404, never a real export)."""
        return export(take, dry=True)

    @app.post("/api/takes/<take>/export")
    def export(take, dry: bool = False):
        _take_or_404(take)
        d = body()
        cap = d.get("capture") or takes.last_capture(take)
        if not cap or not CAPTURE_RE.match(str(cap)):
            raise ApiError(422, "invalid", "capture name: letters, digits, _ and - (e.g. 05_LarissaKadu)")
        argv = ["-m", "dancecap.pipeline", "export", take, "--capture", cap]  # demo_export + triggers.json
        sd = d.get("smplx_dir")  # default (omitted): the take's motion, takes/<take>.toml [export] smplx_dir else smplx/
        if sd:
            if sd != "smplx":  # smplx_detail (hands refinement) is retired (2026-10-07)
                raise ApiError(422, "invalid", "smplx_dir: smplx (or omit it for the take's [export] smplx_dir)")
            argv += ["--smplx-dir", sd]
        if d.get("audio"):
            if d["audio"] not in ("auto", "studio", "phone"):
                raise ApiError(422, "invalid", "audio: auto | studio | phone")
            argv += ["--audio", d["audio"]]
        if d.get("room"):
            argv.append("--room")
        if dry:
            # writes nothing: the commands, the inputs found / missing, the folders it would replace, the Unity lock
            # and the triggers it would carry (python -m dancecap.pipeline export --dry-run). A dry_run run never
            # counts as a run of the demo_export stage.
            r = runs.start("dry_run", argv + ["--dry-run"], take=take, stages=["demo_export"])
            return ok({"run": r, "dry_run": True}, 202)
        lk = _locks()["unity"]
        if lk["held"] and not d.get("override_lock"):
            raise ApiError(423, "locked", "Unity work is in progress (work/unity.lock)", {"lock": "unity",
                                                                                          "owner": lk["owner"]})
        r = runs.start("export", argv, take=take, stages=["demo_export"])
        return ok({"run": r}, 202)

    def review_dir(take: str) -> Path:
        return (paths.take_work(take) / "review").resolve()

    @app.get("/api/takes/<take>/review")
    def review_list(take):
        _take_or_404(take)
        rd = review_dir(take)
        out = []
        if rd.exists():
            for p in sorted(rd.rglob("*")):
                if p.is_file() and p.suffix.lower() in REVIEW_EXT:
                    st = p.stat()
                    out.append({"name": p.relative_to(rd).as_posix(), "mtime": st.st_mtime, "bytes": st.st_size,
                                "kind": "video" if p.suffix.lower() == ".mp4" else "data" if p.suffix.lower() in
                                (".json", ".txt") else "image"})
        out.sort(key=lambda x: -x["mtime"])
        return ok(out[:500])

    @app.get("/api/takes/<take>/review/<path:name>")
    def review_get(take, name):
        _take_or_404(take)
        rd = review_dir(take)
        p = (rd / name).resolve()
        if ".." in Path(name).parts or not str(p).startswith(str(rd) + os.sep) or p.suffix.lower() not in REVIEW_EXT \
                or not p.is_file():
            raise ApiError(404, "not_found", "no such review file")
        return media.file_response(p, request.headers.get("Range"))

    @app.get("/api/takes/<take>/analysis")
    def analysis(take):
        """Classifier and analysis intervals as viewer triggers (counterbalance, floorcraft, move labels)."""
        _take_or_404(take)
        doc = triggers.collect(take)
        fp = paths.take_work(take) / "triggers" / "triggers.json"
        written = read_json(fp, None)
        return ok({"take": take, "time_base": "reference", "lanes": triggers.lanes(doc), "triggers": doc["triggers"],
                   "sources": doc["sources"], "rules": doc["rules"], "threshold": doc["threshold"],
                   "file": {"exists": written is not None, "n": len((written or {}).get("triggers") or []),
                            "path": f"work/{take}/triggers/triggers.json"}}, etag=True)

    @app.put("/api/takes/<take>/triggers")
    def triggers_put(take):
        raise ApiError(501, "not_implemented", "editing the trigger rules from the monitor is phase 2: edit "
                                               "taxonomy/triggers.toml, then run the triggers stage")

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs_list():
        take = request.args.get("take")
        if take and not TAKE_RE.match(take):
            raise ApiError(400, "bad_request", "invalid take")
        lim = _int_arg(request.args, "limit", 50, 1, 200)
        return ok([_run_out(r) for r in runs.list_runs(take=take, active=request.args.get("active") == "1", limit=lim)])

    def run_or_404(rid: str) -> dict:
        if not RUN_RE.match(rid or ""):
            raise ApiError(404, "not_found", "no such run")
        r = runs.describe(rid)
        if r is None:
            raise ApiError(404, "not_found", "no such run")
        return _run_out(r)

    @app.get("/api/runs/<rid>")
    def run_get(rid):
        return ok(run_or_404(rid))

    @app.get("/api/runs/<rid>/log")
    def run_log(rid):
        run_or_404(rid)
        off = _int_arg(request.args, "offset", 0, 0, 1 << 40)
        mb = _int_arg(request.args, "max_bytes", 65536, 1024, 1 << 20)
        return ok(runs.log_chunk(runs.run_dir(rid) / "output.log", off, mb, tail=request.args.get("tail") == "1"))

    @app.post("/api/runs/<rid>/stop")
    def run_stop(rid):
        run_or_404(rid)
        mode = body().get("mode", "now")
        if not isinstance(mode, str):
            raise ApiError(400, "bad_request", "mode: now | after_step")
        try:
            return ok(runs.request_stop(rid, mode))
        except PermissionError as e:
            raise ApiError(409, "conflict", str(e))

    # ------------------------------------------------------------------ runpod
    @app.get("/api/runpod")
    def runpod_get():
        return ok(runpod_summary(), etag=True)

    @app.post("/api/runpod/refresh")
    def runpod_refresh():
        if time.time() - m._last_refresh < 10:
            raise ApiError(429, "rate_limited", "at most one refresh per 10 s")
        m._last_refresh = time.time()
        return ok(runpod_summary(force=True))

    @app.get("/api/runpod/gpus")
    def runpod_gpus():
        if m._gpus and time.time() - m._gpus[0] < 600:
            return ok(m._gpus[1])
        try:
            from ... import remote
            rows = remote.gpu_table()
        except Exception as e:  # noqa: BLE001
            raise ApiError(503, "runpod_unavailable", redact(str(e))[:300])
        m._gpus = (time.time(), rows)
        return ok(rows)

    @app.post("/api/runpod/pods/<pod_id>/delete")
    def pod_delete(pod_id):
        if not POD_ID_RE.match(pod_id or ""):
            raise ApiError(404, "not_found", "no such pod")
        try:
            res = runpod_view.delete_pod(pod_id, str(body().get("confirm") or ""), m.poller)
        except KeyError:
            raise ApiError(404, "not_found", "no such pod")
        except PermissionError as e:
            raise ApiError(403, "forbidden", str(e))
        m.invalidate("runpod")
        return ok(res)

    def orphans() -> list[dict]:
        """dancecap pods flagged orphan by the poller: an open ledger row, no active session, no runner process."""
        rp = runpod_summary(force=True)
        return [{"id": p["id"], "name": p.get("name"), "age_s": p.get("age_s"), "cost_per_hr": p.get("cost_per_hr"),
                 "ledger": p.get("ledger")} for p in rp.get("pods") or []
                if "orphan" in (p.get("flags") or []) and p.get("managed") and not p.get("session")]

    @app.get("/api/runpod/orphans")
    def pod_orphans():
        return ok({"orphans": orphans()})

    @app.post("/api/runpod/cleanup")
    def pod_cleanup():
        """Delete ORPHANED dancecap pods only (never a pod with an active session or a live runner). Body
        {confirm: "DELETE", pods: [ids]}: the ids shown in the confirmation; each must still be an orphan now."""
        d = body()
        if d.get("confirm") != "DELETE":
            raise ApiError(403, "forbidden", "type DELETE to confirm deleting the orphaned dancecap pods")
        now_orphans = {o["id"]: o for o in orphans()}
        want = d.get("pods")
        if want is None:
            want = sorted(now_orphans)  # the dialog without a list: exactly the orphans of this moment
        if not isinstance(want, list) or not all(isinstance(x, str) and POD_ID_RE.match(x) for x in want):
            raise ApiError(400, "bad_request", "pods: a list of pod ids")
        not_orphan = [x for x in want if x not in now_orphans]
        if not_orphan:
            raise ApiError(409, "conflict", f"not orphans (an active session or runner, or gone): {not_orphan}",
                           {"orphans": list(now_orphans.values())})
        from ... import remote
        gone = []
        for pid in want:
            remote._delete(pid, reason="monitor: orphan cleanup (user confirmed)")
            gone.append(pid)
        m.invalidate("runpod")
        return ok({"deleted": gone})

    @app.get("/api/sessions")
    def sessions_list():
        take = request.args.get("take")
        if take and not TAKE_RE.match(take):
            raise ApiError(400, "bad_request", "invalid take")
        lim = _int_arg(request.args, "limit", 50, 1, 200)
        return ok(runpod_view.sessions(take, lim), etag=True)

    def sid_or_404(sid: str) -> None:
        if not SID_RE.match(sid or "") or not (paths.sessions_dir() / sid / "state.json").exists():
            raise ApiError(404, "not_found", "no such session")

    @app.get("/api/sessions/<sid>")
    def session_get(sid):
        sid_or_404(sid)
        return ok(runpod_view.session_snapshot(sid), etag=True)

    @app.get("/api/sessions/<sid>/log")
    def session_log(sid):
        sid_or_404(sid)
        off = _int_arg(request.args, "offset", 0, 0, 1 << 40)
        return ok(runs.log_chunk(paths.sessions_dir() / sid / "runner.log", off,
                                 _int_arg(request.args, "max_bytes", 65536, 1024, 1 << 20),
                                 tail=request.args.get("tail") == "1"))

    @app.post("/api/sessions/<sid>/live-probe")
    def session_live_probe(sid):
        sid_or_404(sid)
        raise ApiError(501, "not_implemented", "the read-only legacy probe (PIPELINE_PROGRESS 4.4) is not wired yet")

    @app.post("/api/sessions/<sid>/stop")
    def session_stop(sid):
        sid_or_404(sid)
        d = body()
        if d.get("mode") != "delete_pod":
            raise ApiError(409, "conflict", "sessions offer only delete_pod (typed confirmation) until stop-now is "
                                            "verified on a fake pod")
        snap = runpod_view.session_snapshot(sid) or {}
        pod_id = (snap.get("summary") or {}).get("pod_id")
        if not pod_id or "delete_pod" not in (snap.get("stop_modes") or []):
            raise ApiError(409, "conflict", "this session has no running pod")
        try:
            res = runpod_view.delete_pod(pod_id, str(d.get("confirm") or ""), m.poller)
        except PermissionError as e:
            raise ApiError(403, "forbidden", str(e))
        except KeyError:
            raise ApiError(404, "not_found", "pod not found")
        m.invalidate("runpod")
        return ok(res)

    @app.get("/api/budget")
    def budget():
        h = request.args.get("plan")
        queued = []
        if h:
            try:
                p = planmod.load(h)
            except planmod.PlanError:
                p = None
            if p is None:
                raise ApiError(404, "not_found", "no such plan")
            queued = p.get("sessions") or []
        return ok(estimate.budget(queued=queued))

    # ------------------------------------------------------------------ unity
    @app.get("/api/unity")
    def unity():
        u = _unity_info()
        u.pop("editor_path", None)
        return ok(u)

    @app.post("/api/unity/open")
    def unity_open():
        u = _unity_info()
        if u["running"]:
            return ok({"running": True, "message": "Unity is already open"})
        if not u["editor_found"]:
            raise ApiError(404, "not_found", f"Unity {u['version']} is not installed under the Hub editor folder")
        subprocess.Popen([str(u["editor_path"]), "-projectPath", str(paths.unity_project())],
                         creationflags=0x00000200 if os.name == "nt" else 0, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
        return ok({"started": True}, 202)

    @app.post("/api/unity/load")
    def unity_load():
        d = body()
        cap = d.get("capture")
        if not cap or not CAPTURE_RE.match(str(cap)):
            raise ApiError(422, "invalid", "capture name required")
        lk = _locks()["unity"]
        if lk["held"] and not d.get("override_lock"):
            raise ApiError(423, "locked", "Unity work is in progress (work/unity.lock): loading a capture would disturb "
                                          "it", {"lock": "unity", "owner": lk["owner"]})
        exe = Path(os.environ.get("LOCALAPPDATA", "")) / "Unity" / "bin" / "unity.exe"
        if not exe.exists():
            raise ApiError(404, "not_found", "the Unity CLI is not installed")
        p = subprocess.run([str(exe), "command", "hm_load", "--name", cap, "--project-path", str(paths.unity_project())],
                           capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW, stdin=subprocess.DEVNULL)
        return ok({"rc": p.returncode, "output": redact((p.stdout or p.stderr or "")[-1000:])},
                  200 if p.returncode == 0 else 409)

    # ------------------------------------------------------------------ logs
    @app.get("/api/logs/<source>")
    def logs(source):
        if source not in ("monitor", "launcher"):
            raise ApiError(404, "not_found", "sources: monitor | launcher")
        p = paths.monitor_dir() / "logs" / f"{source}.log"
        return ok(runs.log_chunk(p, _int_arg(request.args, "offset", 0, 0, 1 << 40),
                                 _int_arg(request.args, "max_bytes", 65536, 1024, 1 << 20),
                                 tail=request.args.get("tail") == "1"))

    # ------------------------------------------------------------------ analyses, live sessions, snapshots (2026-10-21)
    from .. import activity as activity_mod, analyze as analyze_mod, snapshots as snaps_mod

    @app.get("/api/activity")
    def activity_get():
        """Analysis runs, live pod sessions (upload, phase, ETA, cost), recent sessions, budget: the Farm page's
        'Analyses and pod sessions' strip."""
        return ok(m.cached("activity", 2.0, lambda: activity_mod.bundle(rp=runpod_summary())))

    @app.get("/api/analyses")
    def analyses_list():
        now = time.time()
        return ok([{"run": d["run"], "take": d.get("take"), "status": d.get("status"), "created": d.get("created"),
                    "route": (d.get("input") or {}).get("route"), "live": analyze_mod.live(d, now)}
                   for d in analyze_mod.list_runs(30)])

    @app.get("/api/analyses/<run>")
    def analysis_get(run):
        if not analyze_mod.RUN_RE.match(run or ""):
            raise ApiError(404, "not_found", "no such analysis run")
        d = m.cached(f"analysis:{run}", 2.0, lambda: activity_mod.analysis_detail(run))
        if d is None:
            raise ApiError(404, "not_found", "no such analysis run")
        return ok(d)

    @app.get("/api/snapshots")
    def snapshots_list():
        return ok(snaps_mod.list_runs(60))

    @app.get("/api/snapshots/<run>")
    def snapshots_index(run):
        if not snaps_mod.RUN_RE.match(run or ""):
            raise ApiError(404, "not_found", "no such run")
        return ok(snaps_mod.index(run))

    @app.get("/api/snapshots/<run>/<name>")
    def snapshot_file(run, name):
        if not snaps_mod.RUN_RE.match(run or "") or not re.fullmatch(r"\d{2,4}_[a-z0-9_]{1,40}\.(png|html)", name or ""):
            raise ApiError(404, "not_found", "no such snapshot")
        p = snaps_mod.snap_dir(run) / name
        if not p.is_file():
            raise ApiError(404, "not_found", "no such snapshot")
        st = p.stat()
        tag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        if request.headers.get("If-None-Match") == tag:
            return Response(status=304, headers={"ETag": tag})
        return Response(p.read_bytes(), mimetype="image/png" if name.endswith(".png") else "text/html",
                        headers={"ETag": tag, "Cache-Control": "private, max-age=3600"})

    # ------------------------------------------------------------------ farm (defrag map; docs/FARM.md section 7)
    from . import farm_routes
    farm_routes.register(app, m, ok, body, ApiError, runpod_summary)

    return app
