"""Local web server of the annotation tool (Flask app served by waitress on 127.0.0.1 only).

Video: GET/HEAD /video/<camera> with HTTP Range (206 partial content, 416 when unsatisfiable, a multi-range or
malformed header is ignored -> 200) so the browser can seek; only the session's own camera files (or their proxies)
are served. Requests whose Host header is not this loopback address are refused (DNS-rebinding guard) - the videos
and the narration audio are private.

API (JSON):
  GET  /api/session                     session info, cameras, beat grid, proposals, counterbalance, moves, labels,
                                        draft, recogniser status, earlier narrations
  POST /api/labels {segments}           validate + save labels.json (400 + problems when invalid)
  POST /api/validate {segments}         problems only
  POST/DELETE /api/draft                autosave of the working timeline
  POST /api/merge {current, new}        pipeline.merge_segments
  POST /api/narration (multipart)       audio + log (+ settings) -> background job: transcribe + map -> result
  GET  /api/narration/<id>              result.json (or the job state)
  POST /api/narration/<id>/remap        new settings (lag, snap, typesafe), no re-transcription
  POST /api/narration/<id>/transcript   import a transcript JSON [{word, start, end, prob}] for this narration
  GET  /api/narration/<id>/audio        the recorded narration (Range)
  GET  /api/narrations                  list
  POST /api/alias {move, alias}         additions.toml [[moves]] alias + graph rebuild (user accepted in the UI)
  POST /api/propose {name, family, ...} additions.toml proposed move + graph rebuild (user accepted in the UI)
  POST /api/suggest {phrase}            lexicon candidates for a phrase
  POST /api/export                      `python -m dancecap.moves run <take>` (path + fingerprint + Unity export)
  POST /api/proxy/<camera>              H.264 proxy of a camera the browser cannot decode
  GET  /api/jobs/<id>                   background job state
  GET  /api/asr[?refresh=1]             recogniser status
"""
from __future__ import annotations

import json
import mimetypes
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from email.utils import formatdate
from pathlib import Path

from ...take import PACKAGE_ROOT
from . import asr
from .pipeline import NarrationSettings, merge_segments, process
from .session import Session
from .taxonomy_edit import TaxonomyEditError, append_alias, append_proposed_move, rebuild_graph

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
CHUNK = 1 << 20
AUDIO_EXT = {"audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav", "audio/webm": ".webm",
             "audio/ogg": ".ogg", "audio/mp4": ".m4a", "audio/mpeg": ".mp3"}
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("video/mp4", ".mp4")


# ---------------------------------------------------------------------------------------------------------------
# HTTP Range
# ---------------------------------------------------------------------------------------------------------------

def parse_range(header: str | None, size: int):
    """None (no / ignored range: send everything), 'unsatisfiable', or (start, end) inclusive."""
    if not header:
        return None
    m = re.fullmatch(r"\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*", header)
    if not m:
        return None  # multi-range or malformed: a server may ignore Range (RFC 9110 14.2)
    a, b = m.groups()
    if a == "" and b == "":
        return None
    if a == "":
        n = int(b)
        if n == 0 or size == 0:
            return "unsatisfiable"
        return max(0, size - n), size - 1
    start = int(a)
    if start >= size:
        return "unsatisfiable"
    end = int(b) if b else size - 1
    if end < start:
        return None
    return start, min(end, size - 1)


def file_response(path: Path, range_header: str | None, head: bool = False, mimetype: str | None = None):
    from flask import Response
    path = Path(path)
    st = path.stat()
    size = st.st_size
    mime = mimetype or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    headers = {"Accept-Ranges": "bytes", "Content-Type": mime, "Last-Modified": formatdate(st.st_mtime, usegmt=True),
               "Cache-Control": "no-cache", "ETag": f'"{st.st_mtime_ns:x}-{size:x}"'}
    r = parse_range(range_header, size)
    if r == "unsatisfiable":
        headers["Content-Range"] = f"bytes */{size}"
        headers["Content-Length"] = "0"
        return Response(b"", status=416, headers=headers)
    start, end = r if r else (0, size - 1)
    length = max(0, end - start + 1)
    if r:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    headers["Content-Length"] = str(length)
    status = 206 if r else 200
    if head:
        resp = Response(b"", status=status, headers=headers)
        resp.headers["Content-Length"] = str(length)  # the length a GET would send (RFC 9110 9.3.2)
        return resp

    def gen():
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                left -= len(chunk)
                yield chunk

    resp = Response(gen(), status=status, headers=headers, direct_passthrough=True)
    resp.headers["Content-Length"] = str(length)
    return resp


# ---------------------------------------------------------------------------------------------------------------
# background jobs
# ---------------------------------------------------------------------------------------------------------------

class Jobs:
    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def start(self, kind: str, fn, *args, **kw) -> str:
        jid = f"{kind}-{uuid.uuid4().hex[:8]}"
        with self.lock:
            self.jobs[jid] = {"id": jid, "kind": kind, "state": "running", "started": time.time(), "log": []}

        def logf(msg: str) -> None:
            with self.lock:
                self.jobs[jid]["log"].append(str(msg))
            print(msg, flush=True)

        def run():
            try:
                res = fn(*args, log=logf, **kw)
                with self.lock:
                    self.jobs[jid].update(state="done", result=res, ended=time.time())
            except Exception as e:  # noqa: BLE001 - reported to the UI
                traceback.print_exc()
                with self.lock:
                    self.jobs[jid].update(state="error", error=f"{type(e).__name__}: {e}", ended=time.time())

        threading.Thread(target=run, daemon=True, name=jid).start()
        return jid

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(jid)
            return json.loads(json.dumps(j, default=str)) if j else None


# ---------------------------------------------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------------------------------------------

def create_app(session: Session, allowed_hosts: set[str] | None = None, check_asr: bool = True):
    from flask import Flask, abort, jsonify, request, send_from_directory

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
    jobs = Jobs()
    state = {"asr": None}
    edit_lock = threading.Lock()

    def asr_status(refresh: bool = False) -> dict:
        if refresh or state["asr"] is None:
            state["asr"] = asr.engines(refresh=refresh)
        return state["asr"]

    if check_asr:
        threading.Thread(target=asr_status, daemon=True).start()

    @app.before_request
    def guard():
        if allowed_hosts is not None and request.host not in allowed_hosts:
            abort(403)

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        if request.path.startswith("/api/") or request.path in ("/", "/index.html"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def body() -> dict:
        d = request.get_json(silent=True)
        return d if isinstance(d, dict) else {}

    def settings_of(d: dict | None) -> NarrationSettings:
        s = NarrationSettings.from_dict({**session.settings, **(d or {})})
        s.annotator = s.annotator or session.annotator
        return s

    def ndir_of(nid: str) -> Path:
        if not re.fullmatch(r"[0-9A-Za-z_-]{1,64}", nid or ""):
            abort(404)
        d = session.narration_dir / nid
        if not (d / "log.json").exists():
            abort(404)
        return d

    # -- pages ------------------------------------------------------------------------------------------------
    @app.get("/")
    def index():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(STATIC, name)

    @app.route("/video/<cam>", methods=["GET", "HEAD"])
    def video(cam):
        if cam not in session.cameras:
            abort(404)
        p = session.cameras[cam].serve_path()
        if not p.exists():
            abort(404)
        return file_response(p, request.headers.get("Range"), head=request.method == "HEAD")

    # -- session ----------------------------------------------------------------------------------------------
    @app.get("/api/session")
    def api_session():
        info = session.info()
        info.update(moves=session.moves_for_ui(), labels=session.load_working(), draft=session.load_draft(),
                    asr=state["asr"], narrations=session.narrations(),
                    defaults={**NarrationSettings().__dict__, **session.settings})
        return jsonify(info)

    @app.get("/api/moves")
    def api_moves():
        return jsonify(session.moves_for_ui())

    @app.get("/api/asr")
    def api_asr():
        st = asr_status(refresh=request.args.get("refresh") == "1")
        return jsonify({**st, "help": None if st.get("faster-whisper", {}).get("available") else asr.setup_help()})

    # -- labels -----------------------------------------------------------------------------------------------
    @app.post("/api/validate")
    def api_validate():
        return jsonify({"problems": session.validate(body().get("segments") or [])})

    @app.post("/api/labels")
    def api_labels():
        r = session.save(body().get("segments") or [])
        if r.get("ok"):
            session.clear_draft()
        return jsonify(r), (200 if r.get("ok") else 400)

    @app.route("/api/draft", methods=["POST", "DELETE"])
    def api_draft():
        if request.method == "DELETE":
            session.clear_draft()
            return jsonify({"ok": True})
        p = session.save_draft(body().get("segments") or [])
        return jsonify({"ok": True, "path": str(p)})

    @app.post("/api/merge")
    def api_merge():
        d = body()
        return jsonify({"segments": merge_segments(d.get("current") or [], d.get("new") or [])})

    # -- narration --------------------------------------------------------------------------------------------
    def run_process(ndir: Path, settings: NarrationSettings, retranscribe: bool, log=print):
        return process(ndir, session.context(), session.lexicon, settings, retranscribe=retranscribe, log=log)

    @app.post("/api/narration")
    def api_narration():
        f = request.files.get("audio")
        if f is None:
            abort(400, "audio missing")
        try:
            log_doc = json.loads(request.form.get("log") or "{}")
            sett = json.loads(request.form.get("settings") or "{}")
        except ValueError:
            abort(400, "bad log / settings JSON")
        ndir = session.new_narration_dir()
        ext = AUDIO_EXT.get((f.mimetype or "").split(";")[0].strip(), Path(f.filename or "").suffix or ".webm")
        f.save(str(ndir / f"audio{ext}"))
        log_doc.setdefault("camera_offsets", {c.id: {"slope": c.slope, "offset": c.offset}
                                              for c in session.cameras.values()})
        (ndir / "log.json").write_text(json.dumps(log_doc, indent=1), encoding="utf-8")
        s = settings_of(sett)
        (ndir / "settings.json").write_text(json.dumps(s.__dict__, indent=1), encoding="utf-8")
        jid = jobs.start("narration", run_process, ndir, s, True)
        return jsonify({"id": ndir.name, "job": jid})

    @app.get("/api/narration/<nid>")
    def api_narration_get(nid):
        d = ndir_of(nid)
        if (d / "result.json").exists():
            return jsonify(json.loads((d / "result.json").read_text(encoding="utf-8")))
        return jsonify({"id": nid, "state": "pending"}), 202

    @app.post("/api/narration/<nid>/remap")
    def api_remap(nid):
        d = ndir_of(nid)
        if not (d / "transcript.json").exists():
            return jsonify({"error": "no transcript yet"}), 409
        s = settings_of(body().get("settings"))
        if s.typesafe:
            return jsonify({"job": jobs.start("remap", run_process, d, s, False)})
        return jsonify(run_process(d, s, False))

    @app.post("/api/narration/<nid>/transcript")
    def api_transcript(nid):
        d = ndir_of(nid)
        try:
            tr = asr.load_transcript_json(request.get_json(silent=True))
        except asr.ASRError as e:
            return jsonify({"error": str(e)}), 400
        (d / "transcript.json").write_text(json.dumps(tr, indent=1, ensure_ascii=False), encoding="utf-8")
        return jsonify(run_process(d, settings_of(body().get("settings")), False))

    @app.route("/api/narration/<nid>/audio", methods=["GET", "HEAD"])
    def api_narration_audio(nid):
        d = ndir_of(nid)
        a = next((p for p in sorted(d.glob("audio.*"))), None) or (d / "audio16k.wav")
        if not a.exists():
            abort(404)
        return file_response(a, request.headers.get("Range"), head=request.method == "HEAD")

    @app.get("/api/narrations")
    def api_narrations():
        return jsonify(session.narrations())

    # -- vocabulary (user-accepted additions) ------------------------------------------------------------------
    def taxonomy_change(fn, *args):
        with edit_lock:
            try:
                r = fn(*args, graph=session.graph, additions=session.additions_path, zouk1=session.zouk1_path,
                       vocabulary=session.vocabulary_path)
            except TaxonomyEditError as e:
                return jsonify({"ok": False, "error": str(e)}), 400
            if r.get("graph") is not None:
                g = rebuild_graph(session.graph_path, session.additions_path, session.zouk1_path,
                                  session.vocabulary_path, built=r["graph"])
                session.reload_graph(g)
            return jsonify({"ok": True, "message": r.get("message"), "slug": r.get("slug"),
                            "changed": r.get("changed"), "moves": session.moves_for_ui()})

    @app.post("/api/alias")
    def api_alias():
        d = body()
        return taxonomy_change(append_alias, d.get("move"), d.get("alias"))

    @app.post("/api/propose")
    def api_propose():
        return taxonomy_change(append_proposed_move, body())

    @app.post("/api/suggest")
    def api_suggest():
        d = body()
        c = session.lexicon.candidates(d.get("phrase") or "", d.get("previous"))
        return jsonify([x.to_dict() for x in c])

    # -- export / proxies / jobs ------------------------------------------------------------------------------
    def run_export(log=print):
        cmd = [sys.executable, "-m", "dancecap.moves", "run", session.name]
        log("$ " + " ".join(cmd))
        r = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), capture_output=True, text=True, timeout=1800,
                           encoding="utf-8", errors="replace")
        return {"returncode": r.returncode, "stdout": r.stdout[-8000:], "stderr": r.stderr[-4000:]}

    @app.post("/api/export")
    def api_export():
        if session.mode != "take":
            return jsonify({"error": "export needs a take (labels in reference time); this is a lone video"}), 400
        if not session.labels_path.exists():
            return jsonify({"error": "save the labels first"}), 409
        return jsonify({"job": jobs.start("export", run_export)})

    @app.post("/api/proxy/<cam>")
    def api_proxy(cam):
        if cam not in session.cameras:
            abort(404)
        return jsonify({"job": jobs.start("proxy", lambda log=print: str(session.make_proxy(cam, log=log)))})

    @app.get("/api/jobs/<jid>")
    def api_job(jid):
        j = jobs.get(jid)
        if j is None:
            abort(404)
        return jsonify(j)

    app.config["session"] = session
    app.config["jobs"] = jobs
    return app


def free_port(port: int, tries: int = 20) -> int:
    for p in range(port, port + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise OSError(f"no free port in {port}..{port + tries - 1}")


def serve(session: Session, port: int = 8765, open_browser: bool = True, threads: int = 8) -> None:
    from waitress import serve as wserve
    port = free_port(port)
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    app = create_app(session, allowed_hosts=hosts)
    url = f"http://127.0.0.1:{port}/"
    print(f"[annotate] {session.mode} {session.name!r} ({session.time_base} time) -> {session.labels_path}")
    print(f"[annotate] open {url}   (local only; Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    wserve(app, host="127.0.0.1", port=port, threads=threads, channel_timeout=600, ident="dancecap-annotate")
