# Pipeline Monitor: technology, double-click launch, UI and API contract (`dancecap.monitor/1`)

> Research stage "gui" of the pipeline-monitor workflow (2026-10-07). **Status (2026-10-11): implemented** (`dancecap/pipeline/monitor/`; see the integration addendum and section 11). It was written as
> the shared contract for the backend builder (`dancecap/dancecap/pipeline/monitor/`, the run wrapper, the
> launcher) and the frontend builder (`.../monitor/static/`). Companion docs:
> - `docs/PIPELINE_PROGRESS.md` owns the progress events, the ETA and the budget maths. This doc only exposes them.
> - `docs/PIPELINE.md` and `dancecap/pipeline/stages_draft.json` (stage "dag") own the stage registry, the variants
>   and the camera-type behaviour of each stage.
>
> Compute policy (user): heavy CV and neural work runs on Runpod. The local PC (4 logical CPUs, a shared 2080 Ti) does
> orchestration, the UI and light checks. The monitor shows existing pods **read-only** unless the user clicks and
> confirms.

## 0. Decisions on one screen

1. **Shell.** A local web UI shown in a Microsoft Edge `--app` window, with no address bar and its own Edge profile
   in `%LOCALAPPDATA%\dancecap\monitor-edge`. The default browser is the fallback.
   - Edge 154 and the WebView2 runtime 154 are both installed here, so the engine is the same as WebView2.
   - pywebview, Qt and Electron are rejected for v1 (section 2).
2. **Backend.** Flask and waitress in the existing `.venv-nopo4d`, which already has Flask 3.1.3, waitress 3.0.2 and
   psutil 7.2.2. This is the same pattern as the narration tool. **No new dependencies and no pin changes.** It binds
   127.0.0.1 only.
3. **Frontend.** Static HTML with vanilla ES modules, SVG and CSS.
   - No build step, no CDN, no external request (enforced by CSP).
   - Files are served from disk, so an edit plus a reload updates the UI.
4. **Executable.** `tools/PipelineMonitor/PipelineMonitor.exe` is a 6 KB trampoline, built locally by the C#
   compiler that ships with .NET Framework 4.8. That needs no admin, no toolchain and no PyInstaller.
   - It reads `launcher.json` and runs `python.exe -m dancecap.pipeline monitor --launch` without a window.
   - All the logic lives in Python, so updates never need an exe rebuild.
   - Other entry points: Desktop and Start-menu shortcuts, a `.cmd` fallback, `head-movement/Tools/PipelineMonitor.cmd`,
     and the Unity menu item **Head Movement/Pipeline Monitor**.
5. **Process model.**
   - The server runs as `python.exe` under `CREATE_NO_WINDOW`, never `pythonw.exe`. Measured: under pythonw, console
     children such as `remote._ctl`'s runpodctl, ssh and ffmpeg each pop a visible console window.
   - Single instance, through an OS file lock. A second double-click only opens another window.
   - Work runs in detached processes registered on disk (`work/pipeline/runs/`). Closing the window or restarting
     the server never stops a stage or a pod session.
6. **Read-only by default.**
   - Pods, sessions and runs that something else started are only displayed.
   - Creating a pod is always two steps: **plan** (no side effects, returns a `plan_hash` and p50/p90 cost), then
     **run** with `{plan_hash, approve: {pods, max_cost_usd}}`.
   - The server re-checks the budget gate from `PIPELINE_PROGRESS.md` section 8.3 and the consent gate.
7. **Camera types.** The user sets one type per video. One JSON shape (`CameraType`) and one module
   (`dancecap.camera_types`) serve three places:
   - the new-capture wizard;
   - the per-take **Video Review** screen;
   - a camera panel in the narration tool (`python -m dancecap.moves annotate`).

   The type is stored in `takes/<take>.toml [cameras."NN"]`. The automatic suggestion is pre-filled, and the user
   confirms it. Before saving, the UI shows which done stages a change would make stale.
8. **Refresh.** The UI polls one aggregated `GET /api/state` every 2 s, with an ETag. Detail endpoints are polled
   only while their view is visible. Server-sent events are an optional later addition.
9. **Security.**
   - Host-header allowlist and an `Origin` check.
   - A per-instance token header on every write, plus a `SameSite=Strict` cookie for media.
   - No CORS (`flask-cors` is installed but must not be used).
   - Only allowlisted commands: the client never sends an argv.
   - Media is served only for registered takes or picked-file handles.
   - The monitor never reads the Runpod key itself (only `remote._ctl` does, in its child's environment).
   - Pod JSON is reduced to allowlisted fields before it reaches the UI.

## 1. Context, measured on this PC (2026-10-07)

| fact | value | consequence |
|---|---|---|
| OS, shells | Windows 10 Home 19045; Windows PowerShell 5.1.19041 only (no `pwsh` on PATH); .NET Framework 4.8 (release 533325) | `.ps1` helpers must be 5.1-safe (section 3.6) |
| C# compiler | `%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe` 4.8.9232 (C# 5) | Compiled a 6 KB WinForms trampoline in the scratchpad; `JavaScriptSerializer` (System.Web.Extensions) parsed a `launcher.json` and correctly refused backslash paths in invalid JSON |
| Edge / WebView2 | Edge stable at `C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe`; WebView2 runtime 154.0.4258.62 (HKLM EdgeUpdate client `{F3017226-...}`) | `--app` window available on every Windows 10 PC with Edge |
| Python | `.venv-nopo4d`: Python 3.11.16, `include-system-site-packages = true` on the conda `mamma` env; Flask 3.1.3, waitress 3.0.2, Werkzeug 3.1.8, flask-cors 6.0.5, psutil 7.2.2, tkinter. **Not installed:** pywebview, pythonnet, Qt, tomlkit, pywin32 | No new packages needed; TOML edits need a small owned-block editor (section 7.4) |
| venv launcher | `Scripts\python(w).exe` is a redirector. `Popen().pid` was 32824 while the interpreter's `os.getpid()` was 34792 (its parent) | Process records must store the pid that the interpreter writes itself (section 4.2) |
| pythonw | `sys.stdout is None`. A console child started the way `remote._ctl` starts one got **its own visible console window** (`GetConsoleWindow()` non-zero, visible) | Never run the server under pythonw |
| python.exe + `CREATE_NO_WINDOW` | The same child shares a **windowless** console (`hwnd 0`, 4 processes on the console) | This is the server's process mode |
| startup cost | `import flask, waitress` takes 1.42 s; a pythonw spawn-to-exit takes 1.9 s | First window in about 3 s; a second launch against a running server takes about 1 s |
| tools | ffmpeg/ffprobe 8.1.1 on PATH; runpodctl in `~/.local/bin`; Unity 6000.6.3f1 under `C:\Program Files\Unity\Hub\Editor`; Unity CLI in `%LOCALAPPDATA%\Unity\bin\unity.exe` | Thumbnails and proxies run locally; "Open Unity" runs the Hub editor |
| UI precedent | `dancecap/dancecap/moves/annotate/server.py`: Flask + waitress on 127.0.0.1:8765, Host guard, HTTP Range video (`file_response`), a `Jobs` thread pool, vanilla `static/app.js` | Reuse `file_response` / `parse_range` and the Host guard; launch the narration tool as a separate process |
| locks | `work/gpu.lock` is a file holding `"<pid> <label>"` (`dancecap.gpu.exclusive`, stale pid taken over). `work/unity.lock` is a **directory** with `owner.txt`, held by Unity workflows | The monitor shows both and honours them before local GPU or Unity work |
| Runpod now | ledger 36 rows, **0 open**, spent $19.94 of $30; newest session `20261007-065744-batch` ended ok | The Runpod panel shows "no pods running" |
| workspace .gitignore | ignores `/*` except allowlisted dirs (`dancecap/`, `prototype/`, `atlas_bridge/`) | `tools/` needs a `!/tools/` line, and the built exe stays ignored |

## 2. Technology decision

### 2.1 Window ("shell")

| option | new deps | microphone (narration) | startup | look | update without rebuild | verdict |
|---|---|---|---|---|---|---|
| **Edge `--app` window + local Flask** | none | **yes**: Edge's permission prompt; 127.0.0.1 is a secure context; the grant persists in the dedicated profile, per origin | about 1 s after the server | own window, no address bar, Edge's engine and devtools (F12) | yes | **chosen** (`shell: "edge-app"`) |
| pywebview (WinForms + WebView2 through pythonnet) | pywebview, pythonnet, clr-loader, cffi, in a separate venv because `.venv-nopo4d` pins must not move | **uncertain**: its `edgechromium.py` registers no `PermissionRequested` handler, and getUserMedia problems are reported (issue #909) | adds the CLR load | native title and icon | yes | later option `shell: "pywebview"`, same URL |
| default browser tab | none | yes | fast | a tab among others; the user's extensions run on the page | yes | fallback (`shell: "browser"`) |
| Qt (PySide6) | 100+ MB | QtMultimedia | slower | native widgets | yes | rejected: the UI would be rewritten, and the narration tool's web UI could not be reused |
| Electron / Tauri | Node or Rust toolchain | yes | ok | native | needs a rebuild | rejected |

**Edge window details** (`monitor/shell.py`):
- **Path.** Read `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe`, then fall back to the two
  Program Files locations.
- **Arguments:**
  ```
  msedge.exe --app=http://127.0.0.1:<port>/#<route>
             --user-data-dir=%LOCALAPPDATA%\dancecap\monitor-edge
             --no-first-run --no-default-browser-check --window-size=1600,1000
  ```
- **Why a separate profile.** It gives a separate Edge process tree, so the user's own browsing profile is not
  touched and their extensions never run on the page. The microphone permission and the window bounds are kept
  per profile.
- **More windows.** The narration tool and pop-out logs open in the same profile, so they become more app windows
  of the same Edge instance.
- **Caveat.** The narration tool's origin is `127.0.0.1:8765`. If its `free_port()` falls back to another port, the
  origin changes and Edge asks for the microphone again.

### 2.2 Packaging and launch

| option | antivirus / SmartScreen | admin | size | update | verdict |
|---|---|---|---|---|---|
| **C# trampoline built locally with the in-box `csc.exe`** | tiny, no packed payload, no Mark-of-the-Web (built on this PC), so no SmartScreen prompt | no | 6 KB | reads `launcher.json` on every start; all logic is in Python | **chosen** |
| PyInstaller `--onefile` | frequent Defender ML false positives on onefile bootloaders (`Trojan:Win32/Wacatac.*!ml`); unpacks to `%TEMP%` on every start | no | 10-30 MB | rebuild for every change to frozen code | rejected |
| PyInstaller `--onedir` | fewer false positives, but still a second Python | no | 20-60 MB | rebuild | rejected: the backend must import `dancecap` and run stage CLIs inside `.venv-nopo4d` anyway, so a frozen copy would only drift from it |
| `.vbs` | n/a | no | n/a | yes | rejected: VBScript is deprecated and will be disabled by default from about 2027 |
| `.cmd` | flashes a console | no | n/a | yes | fallback only |
| `.lnk` on the Desktop and in the Start menu | none | no | n/a | yes | added; it points at the exe |

## 3. Launch, single instance, lifetime and updates

### 3.1 Files

```
HeadMovement/                                   (workspace repo, private)
  tools/PipelineMonitor/                        needs "!/tools/" in the workspace .gitignore; the built .exe stays ignored
    PipelineMonitor.cs                          trampoline source (C# 5: no $"", no ?., no => members)
    build.ps1                                   csc build -> PipelineMonitor.exe (+ monitor.ico); ASCII, 5.1-safe
    launcher.json                               {python, cwd, args, start_timeout_s, log}; relative to the exe, forward slashes
    PipelineMonitor.cmd                         fallback without the exe
    install_shortcuts.ps1                       Desktop + Start menu "Pipeline Monitor.lnk" (WScript.Shell COM)
    monitor.ico
    PipelineMonitor.exe                         built locally (git-ignored)
  dancecap/dancecap/pipeline/
    __main__.py                                 CLI: monitor | plan | run | probe | ... (shared with the dag/backend stages)
    runwrap.py                                  wrapper of every process the monitor or `pipeline run` starts (section 4.2)
    monitor/__init__.py                         launch() and serve() entry points
    monitor/server.py                           Flask app factory + the routes of section 7
    monitor/shell.py                            Edge/browser window, single instance, server.json
    monitor/pick.py                             native file dialog, run as a short subprocess (tkinter)
    monitor/media.py                            thumbnails and strips (ffmpeg); Range responses through annotate.server.file_response
    monitor/static/                             index.html, app.js, api.js, views/*.js, components/*.js, style.css, icons.svg,
                                                manifest.webmanifest
  dancecap/work/monitor/                        runtime, git-ignored: monitor.lock, server.json, settings.json, logs/, staging/
  dancecap/work/pipeline/runs/<run_id>/         run.json, pid.json, output.log, rc.json, stop.json (section 4)
%LOCALAPPDATA%\dancecap\monitor-edge\           Edge profile (window bounds, microphone grant)

head-movement/                                  (public MIT repo: no secrets, no absolute paths, no dancer or take names)
  Assets/Editor/PipelineMonitorMenu.cs          Head Movement/Pipeline Monitor (+ ".../Pipeline Monitor Location...")
  Tools/PipelineMonitor.cmd                     relative launcher: ..\..\tools\PipelineMonitor\PipelineMonitor.exe
```

### 3.2 Double-click sequence

```
PipelineMonitor.exe  (also reached from the .lnk, Tools\PipelineMonitor.cmd, or the Unity menu)
  1. Mutex Local\dancecap-pipeline-monitor-launch: two quick double-clicks give one launch. The second waits
     up to 15 s, then exits.
  2. Read launcher.json next to the exe and resolve paths against the exe folder. Python missing -> MessageBox
     naming the path and the json file.
  3. Start "<python.exe> -m dancecap.pipeline monitor --launch [forwarded args]" with cwd = dancecap/ and
     CreateNoWindow = true. The exe forwards only its own args (--take T, --view V, --capture C); Python validates them.
  4. Wait up to start_timeout_s:
       rc 0           -> exit
       rc != 0        -> MessageBox with the last 20 lines of launcher.log and its path
       still running  -> MessageBox "slow start, see <log>"

python -m dancecap.pipeline monitor --launch        (stdlib imports only: fast)
  a. Read server.json. If its pid is alive (psutil, matching create_time) and GET /api/health returns the same
     `instance` within 1 s: open a window at the route, exit 0.
  b. Otherwise spawn "<python.exe> -m dancecap.pipeline monitor --serve" with
     CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, stdin/stdout/stderr = DEVNULL, close_fds.
     The server writes its own log. It is never DETACHED_PROCESS (see 3.6).
  c. Wait at most 20 s for server.json plus /api/health, open the window, exit 0. On failure exit 1 with the
     reason in launcher.log.

python -m dancecap.pipeline monitor --serve
  - First, point sys.stdout and sys.stderr at work/monitor/logs/monitor.log (a rotating handler). Nothing is ever
    printed to a console.
  - Take work/monitor/monitor.lock with msvcrt.locking(LK_NBLCK) on a handle held for the whole process. The OS
    releases it when the process dies, so a stale lock cannot exist. If the lock is busy: exit 3.
  - Port: the first free one in 8790..8799. Generate `instance` (uuid4) and `token` (secrets.token_urlsafe(32)).
    Write server.json {pid, port, instance, started, version, static_rev}. The token is never written to disk.
  - waitress.serve(host="127.0.0.1", port, threads=16, channel_timeout=120, ident="dancecap-monitor").
  - Exit after idle_exit_min (default 20) without any API request. Runs are separate processes and keep going.
    A hidden window still polls every 10 s, so it does not count as idle. A watchdog gap of more than 60 s (the PC
    slept) restarts the idle timer, so the server does not exit the moment the PC wakes under an open window.
```

**Startup budget.** About 0.4 s to spawn the process. Flask and waitress imports take 1.4 s (measured). The Edge
window takes about 1 s. **The target is 3 s or less to first paint**, and about 1 s when a server is already
running.
- The server process must not import numpy, torch, cv2 or stage modules at import time. Anything that needs them
  runs in a run process, or is imported lazily inside the handler.

### 3.3 `launcher.json` (read on every start, so editing it needs no rebuild)

```json
{
 "python": "../../.venv-nopo4d/Scripts/python.exe",
 "cwd": "../../dancecap",
 "args": ["-m", "dancecap.pipeline", "monitor", "--launch"],
 "start_timeout_s": 30,
 "log": "../../dancecap/work/monitor/logs/launcher.log"
}
```

- Use forward slashes: a hand-edited backslash path is invalid JSON (verified).
- Python reads every config with `encoding="utf-8-sig"`, because PS 5.1's `Set-Content -Encoding UTF8` writes a BOM.
- Never point `python` at `envs\mamma\python.exe`: its numpy crashes when that interpreter is launched directly.

### 3.4 Unity menu item and the head-movement launcher

`head-movement/Assets/Editor/PipelineMonitorMenu.cs` (Editor-only; C# 9 is fine in Unity 6):

```csharp
using System; using System.Diagnostics; using System.IO; using UnityEditor; using UnityEngine;

public static class PipelineMonitorMenu
{
    const string PrefKey = "HeadMovement.PipelineMonitor.Path";   // per-user EditorPrefs (HKCU registry), never in the repo
    const string EnvKey  = "HEADMOVEMENT_PIPELINE_MONITOR";

    [MenuItem("Head Movement/Pipeline Monitor", priority = 0)]
    public static void Open()
    {
        string exe = Resolve();
        if (exe == null)
        {
            if (EditorUtility.DisplayDialog("Pipeline Monitor",
                    "The Pipeline Monitor launcher was not found next to this project (../tools/PipelineMonitor/).",
                    "Locate...", "Cancel")) Locate();
            return;
        }
        var psi = new ProcessStartInfo(exe) { UseShellExecute = true, WorkingDirectory = Path.GetDirectoryName(exe) };
        // optional: when a capture is loaded in Play mode, pass it so the monitor opens that take ("--capture <name>")
        Process.Start(psi);   // never waits: the editor stays responsive
    }

    [MenuItem("Head Movement/Pipeline Monitor Location...", priority = 1)]
    public static void Locate()
    {
        string p = EditorUtility.OpenFilePanel("Pipeline Monitor launcher", Path.GetDirectoryName(Resolve() ?? ""), "exe,cmd");
        if (!string.IsNullOrEmpty(p)) EditorPrefs.SetString(PrefKey, p);
    }

    static string Resolve()
    {
        string root = Path.GetFullPath(Path.Combine(Application.dataPath, "../.."));   // the workspace holding the project
        foreach (var p in new[] { EditorPrefs.GetString(PrefKey, ""), Environment.GetEnvironmentVariable(EnvKey) ?? "",
                                  Path.Combine(root, "tools/PipelineMonitor/PipelineMonitor.exe"),
                                  Path.Combine(root, "tools/PipelineMonitor/PipelineMonitor.cmd") })
            if (!string.IsNullOrEmpty(p) && File.Exists(p)) return p;
        return null;
    }
}
```

`head-movement/Tools/PipelineMonitor.cmd`. It is ASCII only and runs only the exe, because non-ASCII paths are
fragile in cmd.exe:

```bat
@echo off
rem Starts the dancecap Pipeline Monitor from the workspace that holds this project.
setlocal
set "EXE=%HEADMOVEMENT_PIPELINE_MONITOR%"
if not defined EXE set "EXE=%~dp0..\..\tools\PipelineMonitor\PipelineMonitor.exe"
if not exist "%EXE%" (
  echo Pipeline Monitor not found: "%EXE%"
  echo Build it: powershell -NoProfile -ExecutionPolicy Bypass -File tools\PipelineMonitor\build.ps1
  pause
  exit /b 1
)
start "" "%EXE%" %*
```

**Mapping a capture back to a take** (`--capture`). The monitor scans
`work/*/export/<capture>_demo_export.json`. The newest match picks the take, and the window opens on that take.

### 3.5 Build and shortcuts (once, no admin)

```
powershell -NoProfile -ExecutionPolicy Bypass -File tools\PipelineMonitor\build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File tools\PipelineMonitor\install_shortcuts.ps1
```

`build.ps1` runs:
```
csc -nologo -target:winexe -optimize+ -win32icon:monitor.ico -out:PipelineMonitor.exe
    -r:System.Windows.Forms.dll -r:System.Web.Extensions.dll PipelineMonitor.cs
```
- It checks `$LASTEXITCODE`.

`install_shortcuts.ps1`:
- Creates `Pipeline Monitor.lnk` on `[Environment]::GetFolderPath('Desktop')` (which handles OneDrive-redirected
  desktops) and under `Programs\Head Movement\`.
- Uses `(New-Object -ComObject WScript.Shell).CreateShortcut()` with TargetPath, WorkingDirectory and
  `IconLocation = "<exe>,0"`.
- The shortcut can be pinned to the taskbar by hand.

### 3.6 Windows quirks checklist (all builders)

| quirk | rule |
|---|---|
| pythonw has no console. Console grandchildren (runpodctl, ssh, scp, ffmpeg, git) get **visible** console windows; `sys.stdout` is None | The server is `python.exe` + `CREATE_NO_WINDOW` (measured, section 1). Redirect stdout and stderr to the log before anything else. |
| `DETACHED_PROCESS` children have no console, so each console grandchild opens a new window | Never use it. Children get `CREATE_NO_WINDOW \| CREATE_NEW_PROCESS_GROUP` (0x08000000 \| 0x00000200). |
| The venv `python.exe` is a redirector: the Popen pid is not the interpreter pid | `runwrap` writes `pid.json` with `os.getpid()` and `create_time`. Kill trees with psutil from the launcher pid. |
| `GenerateConsoleCtrlEvent` (`os.kill(pid, CTRL_BREAK_EVENT)`) only reaches processes on the caller's console; the server shares none with its runs | Stop runs cooperatively through `stop.json` (section 4.3), never through console control events. |
| A held file handle blocks a writer's `os.replace` (WinError 5/32) | Open, read, close (`PIPELINE_PROGRESS.md` 5.2). The monitor never holds handles on session, ledger or take files. Its own writes go to a temp file, then `os.replace` with 3 retries on `PermissionError`. |
| PS 5.1: `.ps1` double-click opens Notepad; the default ExecutionPolicy is Restricted | `.ps1` files are never entry points. Run them as `powershell -NoProfile -ExecutionPolicy Bypass -File ...` (per process, nothing persistent). |
| PS 5.1 encodings: `Out-File` and `>` write UTF-16LE; `Set-Content -Encoding UTF8` writes a BOM; BOM-less `.ps1` is read as ANSI | Keep `.ps1` files ASCII. Write files with `[IO.File]::WriteAllText($p, $s, (New-Object Text.UTF8Encoding $false))`. Python reads with `utf-8-sig`. |
| PS 5.1 syntax and data: no `&&`, `\|\|`, `??` or ternary; `ConvertTo-Json` defaults to `-Depth 2`; `ConvertFrom-Json` unwraps one-element arrays; native stderr with `2>&1` becomes an ErrorRecord | Keep PowerShell to build and shortcut scripts. Do the logic in Python. |
| Non-ASCII paths (the soundtrack name has an accent) | Never route them through cmd.exe or PowerShell. Python with `subprocess` list argv uses CreateProcessW. |
| Chromium opens at most 6 connections per host:port | The monitor plays at most 2 videos at once; the rest are image strips. Synced multi-view playback belongs to the narration tool, which has its own origin. |
| Browsers never reveal a dropped or picked file's real path | Picking goes through `POST /api/fs/pick`: a native dialog in a short `pick.py` subprocess (tkinter `askopenfilenames`, topmost, own main thread), plus a "paste a folder path" field that the server validates. No drag-and-drop for sources. |
| Edge decodes H.264 and AAC. HEVC needs the Store extension plus hardware | `probe.codec == "hevc"` gives `playable: false` and offers "Make proxy". The proxy is shared with the narration tool at `work/<take>/moves/proxy/<cam>.mp4`. |

### 3.7 Logs

| file | writer | content |
|---|---|---|
| `work/monitor/logs/launcher.log` | exe (one line per start) and `--launch` | time, resolved python, rc, chosen port, window shell |
| `work/monitor/logs/monitor.log` (rotating, 5 x 2 MB) | server | requests (method, path, status, ms; **never headers, bodies or query values of media routes**), runpodctl poll results (status only), errors with tracebacks |
| `work/pipeline/runs/<id>/output.log` | the run process (stdout and stderr) | stage output |
| session `runner.log` / `progress/*.jsonl` | `remote_jobs` (unchanged) | see PIPELINE_PROGRESS |

**Redaction.** One filter applies to every log the server writes and every log line the API returns. It masks:
- `rpa_[A-Za-z0-9]+`;
- any run of 32 or more `[A-Za-z0-9_-]` characters that is **not** pure lowercase hex.

Bundle sha256 values and session ids therefore stay readable, while API keys and tokens do not. Log lines reach the
UI as text and are rendered with `textContent` only.

### 3.8 Updating without rebuilding

- **Static files** are served from disk with `Cache-Control: no-store` for `index.html` and an ETag for the rest. F5
  in the window picks up UI changes.
- **Version check.** `GET /api/health.static_rev` is a hash of the static files' mtimes and sizes. When it differs
  from the value the page loaded with, the page shows "UI updated - reload".
- **Backend code.** The menu item **Restart backend** calls `POST /api/restart`. The server spawns
  `--launch --replace` and exits. The page's polling rides over the gap. Runs are unaffected.
- **The exe** changes only if `PipelineMonitor.cs` changes.

## 4. Processes, runs and stop semantics

### 4.1 Run registry (`work/pipeline/runs/<run_id>/`)

Every process that the monitor or `python -m dancecap.pipeline run` starts is a **run**.
- **Run id:** `r-YYYYMMDD-HHMMSS-<4 hex>`, local time like session ids.
- **Files:** `run.json` (written by the starter), `pid.json` and `rc.json` (written by `runwrap`), `output.log`, and
  the optional `stop.json` (written by the monitor).
- **Restart safety.** A server restart re-discovers runs from these files, so nothing depends on server memory.

| kind | what it runs | example |
|---|---|---|
| `pipeline` | the orchestrator `python -m dancecap.pipeline run <take> --plan <plan.json>`: a confirmed plan, stage by stage, respecting locks; pod stages are grouped into the plan's sessions | "Plan & run to demo_export" |
| `stage` | one allowlisted stage CLI from the registry | `python -m dancecap beats <take>` |
| `session` | `python -m dancecap remote run --take <take> --jobs ... --max-cost X --max-minutes M [--gpus ...]` | started by a `pipeline` run or directly |
| `dry_run` | stage dry-runs that have side effects but **never create a pod**: `remote run --dry-run` (builds the bundle and estimate; 141 s for 1.5 GB on 2026-10-07), `front_pod write-job` | the "Dry-run bundle" button |
| `tool` | long-lived servers: `python -m dancecap.moves annotate --take T [--camera C] --no-browser` | "Narrate moves" |
| `probe` | `python -m dancecap.pipeline probe <take>` or staging probes (ffprobe plus the camera-motion suggestion) | wizard, Video Review |
| `export` | `python -m dancecap.demo_export <take> --capture <Name> [...]` | "Export to Unity" |
| `media` | ffmpeg proxies and thumbnail strips | "Make proxy" |
| `external` | (discovered, not started by us) | runner processes seen through session dirs; runs from workflows |

`run.json` (contract):
```json
{"id": "r-20261007-181502-a1b2", "kind": "stage", "take": "larissa_kadu_38", "stages": ["beats"],
 "plan_hash": "sha1:3f2a9c1e...", "origin": "monitor",
 "argv_display": ["python", "-m", "dancecap", "beats", "larissa_kadu_38"],
 "cwd": "dancecap", "created": 1791400502.1, "session_sid": null,
 "env_keys": ["DANCECAP_PROGRESS", "DANCECAP_PROGRESS_JOB", "PYTHONIOENCODING", "PYTHONUNBUFFERED", "PYTHONUTF8"]}
```
- `argv_display` never holds secrets, because none are ever passed as arguments.
- `env_keys` lists the names of the environment variables set for the run, never their values.

### 4.2 `runwrap` (`python -m dancecap.pipeline.runwrap --run-dir <dir> -- <module> [args]`)

- **On start** it writes `pid.json` `{pid: os.getpid(), create_time, started}`, then runs the module with
  `runpy.run_module(..., run_name="__main__")`, with `sys.argv` set.
- **On exit** it writes `rc.json` `{rc, ended, error}` (`error` is the exception type and message, at most 500
  characters). The write happens in a `finally` block and through `atexit`.
- **Stop watcher.** A daemon thread polls `stop.json` every 1 s (section 4.3).
- **Child environment:**
  - `PYTHONIOENCODING=utf-8`, `PYTHONUTF8=1`, `PYTHONUNBUFFERED=1`;
  - `DANCECAP_PROGRESS*` as defined in PIPELINE_PROGRESS 3.1;
  - nothing else is added. The Runpod key is read by `remote._api_key()` inside the run, as today.
- **Flags:**
  - every run: `CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP`, `stdin=DEVNULL`, stdout and stderr appended to
    `output.log`;
  - local heavy runs: `BELOW_NORMAL_PRIORITY_CLASS`, so the UI and Unity stay responsive on 4 CPUs.
- **Run state** is derived from these files:
  - `starting` until `pid.json` exists;
  - `running` while the pid is alive (psutil, matching `create_time`);
  - `done` or `failed` from `rc.json`;
  - `lost` when the pid is gone and there is no `rc.json`;
  - `stopping` or `stopped` when `stop.json` exists;
  - `waiting_human` when a pipeline run exits with rc 3 and a gate note.

### 4.3 Stop semantics

| mode | local `stage`, `export`, `probe`, `media`, `tool` | `pipeline` | `session` started by us (L0 runner) | `session` with the L1 hook | `external` session |
|---|---|---|---|---|---|
| `after_step` | not offered | the orchestrator stops before the next stage | not offered | `remote_jobs` checks `<session>/STOP` between steps (additive L1 patch, PIPELINE_PROGRESS 10.2) | not offered |
| `now` | `stop.json`, then runwrap kills the run's **child** processes (psutil) and calls `_thread.interrupt_main()`, so `finally` blocks run; after a 30 s grace, psutil kills the tree. Outputs follow CONTRACTS (atomic writes, done markers); a dead `gpu.lock` holder is taken over by `gpu.py` | stops the current child the same way, then exits | same mechanism: killing the ssh child unblocks the runner, then `KeyboardInterrupt` reaches `pod_session`'s `finally`, which deletes the pod and closes the ledger row. **Must be verified on a dry-run session and a fake pod before it is offered for real pods** | same | not offered |
| `delete_pod` | n/a | n/a | `remote._delete(pod_id)` after a typed confirmation. The runner's next ssh fails, and its `finally` and `_finish` run | same | **offered** (with the typed confirmation); this is the only action on runners we did not start |

If a hard kill ever happens (a tree kill after the grace period, or a crash), the pod lives until its on-pod kill
switch fires. The Runpod panel then shows it red: "orphan: deletes itself at HH:MM ($X)", with a [Delete now] button.

### 4.4 Local compute policy (enforced by the orchestrator, shown in the UI)

| resource | limit | source of truth |
|---|---|---|
| local CPU analyses (`runs_on: local`, e.g. ingest, cam_map, song, timing, physics, demo_export) | **2 at a time** (PIPELINE.md section 5: 4 logical CPUs, with headroom for the monitor and Unity). A stage whose registry entry sets `heavy: true` runs alone | the run registry |
| UI helper work (picker probes, thumbnails, strips, proxies for the open screen) | 1 extra slot, so the UI never waits behind analyses | the run registry |
| local GPU runs (`runs_on: local_gpu`) | 1, through `work/gpu.lock` (owner shown) | `dancecap.gpu` |
| Unity runs (`runs_on: unity`) and exports while Unity work is in progress | refused with 423 while `work/unity.lock/` exists, unless the user overrides after a confirmation | `work/unity.lock/owner.txt` |
| pod sessions | 1 at a time by default. A second needs the launch gate with every running pod's worst case (PIPELINE_PROGRESS 8.3) | ledger + plan |
| human stages (`runs_on: human`) | the orchestrator pauses (`waiting_human`) and the UI says what is needed | registry gates |

**Where each stage runs** is the registry's `runs_on`. The UI never moves a heavy stage onto the local PC. A "Run
here instead" button appears only when the registry declares an alternative host for that stage.

## 5. Security and privacy (contract)

1. **Binding.** 127.0.0.1 only.
2. **Host header.** It must be one of `127.0.0.1:<port>`, `localhost:<port>` or `[::1]:<port>`, otherwise 403. This
   is the DNS-rebinding guard, as in the narration tool.
3. **Writes** (`POST`, `PUT`, `PATCH`, `DELETE`) require all of:
   - `X-Monitor-Token: <token>`;
   - `Content-Type: application/json`, except the multipart picker;
   - an `Origin` header, when present, equal to `http://127.0.0.1:<port>`.

   **Token delivery.** The token is injected into `index.html` at serve time as
   `<meta name="dancecap-token" content="...">`. A cross-origin page can neither read it nor send a custom header
   without a preflight, and the server answers no preflight.
4. **Reads and media.** `index.html` sets a `dcm_<port>=<random>; HttpOnly; SameSite=Strict; Path=/` cookie (named
   per port: cookies are scoped per host, so another server on 127.0.0.1 cannot overwrite it). Media and API
   GETs require it, so other sites cannot embed or probe our videos. A 403 from the token or cookie check carries
   `details.auth` (`token` | `cookie`); the UI then reloads itself once (a window left open across a server
   restart), as it does when `/api/state` reports a different `monitor.instance` than the one it booted with.
5. **No CORS.** Never enable `flask-cors`. Send no `Access-Control-*` headers.
6. **Response headers:**
   - `Content-Security-Policy: default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:;
     connect-src 'self'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'`;
   - `X-Content-Type-Options: nosniff`;
   - `Referrer-Policy: no-referrer`.
7. **Commands come from the registry only.** The client sends stage ids, takes and options. The server builds the
   argv from templates and validates every value:
   - take `^[a-z0-9][a-z0-9_]{0,47}$`
   - cam `^[0-9]{2}$`
   - capture `^[A-Za-z0-9_-]{1,64}$`
   - GPU names from `remote.gpu_table()` only
   - numbers with bounds
8. **Files.**
   - Media is served only for (a) cameras of a registered take and (b) handles returned by the picker in this server
     instance.
   - Review images only from under `work/<take>/review/`, after path normalisation and an extension allowlist.
   - No file paths in URLs.
9. **Runpod.**
   - The monitor calls `remote.ledger()`, `remote.spent()`, `remote.cap()`, `remote.status()` and
     `remote._ctl("pod", "list")`. The key exists only in `_ctl`'s child environment.
   - `/api/config` reports `runpod_key: "present" | "missing"` and nothing more.
   - Pod JSON is reduced to `id, name, desiredStatus, runtimeStatus, costPerHr, gpu type/count, cloud,
     lastStartedAt`, plus the mapped port of 22/tcp when the live probe is on. Environment fields are dropped.
   - runpodctl error text is redacted (3.7) before display.
10. **Pods are user-only.** The monitor never creates, stops or deletes a pod on its own.
    - Delete is offered only for pods named `dancecap-*` or known to the ledger, after a typed confirmation of the
      pod id's last 4 characters.
    - Pods created elsewhere are listed as "not managed by dancecap" with no action.
11. **Privacy.**
    - Videos, narration audio, portraits, soundtracks and dancer data never leave the PC through the monitor. Pod
      uploads happen only through `remote_jobs` bundles in runs the user approved.
    - The pod launch gate requires `consent.<role>.cloud_processing == true` for every dancer (TMR_SOMA_PLAN G0).
    - The TypeSafe key is never touched. The narration tool's optional TypeSafe toggle keeps its own behaviour.
12. **Public repo.** The Unity menu script and `Tools/PipelineMonitor.cmd` contain no absolute paths, user names,
    take names, dancer names or keys. Path overrides live in EditorPrefs or an environment variable.

## 6. UI sketch

The wireframes are independent snapshots, so their numbers do not add up across sketches. Values come from
PIPELINE_PROGRESS and the stage draft, or are made up for illustration.

### 6.1 Layout, navigation and visual language

```
+------------------------------------------------------------------------------------------------------------------+
| PIPELINE MONITOR   Takes   New capture   Runpod   Runs   Logs             RUNPOD connected  $19.94/$30 [###|..]  |
+------------------------------------------------------------------------------------------------------------------+
| (amber) larissa_kadu_38: cameras 03, 05 not confirmed - budget p90 +14 pod-min (+$0.30)        [Review videos]   |
+-------------------------------+----------------------------------------------------------------------------------+
| TAKES                [+ New]  |                                                                                  |
| larissa_kadu_38  multi  8 cam |                          (route content: 6.2 - 6.11)                             |
|  ######=::......  41/53       |                                                                                  |
|  texture 62%   ETA 4 min      |                                                                                  |
| larissa_kadu     multi  8 cam |                                                                                  |
|  ###############  done        |                                                                                  |
+-------------------------------+----------------------------------------------------------------------------------+
| 1 run active | GPU lock: free | Unity: open, lock held (owner) | progress level L0 | monitor :8790 v1.0.0          |
+------------------------------------------------------------------------------------------------------------------+
```

**Hash routes:**

| route | view |
|---|---|
| `#/` | dashboard |
| `#/new` | new-capture wizard |
| `#/take/<take>` | overview and DAG |
| `#/take/<take>/videos` | video review |
| `#/take/<take>/stage/<id>` | stage inspector |
| `#/take/<take>/runs` | runs of the take |
| `#/take/<take>/analysis` | analysis and triggers |
| `#/runpod` | Runpod panel |
| `#/runpod/session/<sid>` | session view |
| `#/runs/<id>` | one run |
| `#/logs` | logs |
| `#/settings` | settings |

The launcher's `--take T --view videos|overview|runpod|new` maps to these routes.

**Stage status.** Every status has a letter or icon **and** a colour, never colour alone.

| status | colour | strip glyph |
|---|---|---|
| done | green | `#` |
| running | blue, animated bar; static under `prefers-reduced-motion` | `=` |
| queued | blue outline | |
| stale | amber | `:` |
| failed | red | `!` |
| blocked (needs you) | purple | `?` |
| ready | white outline | |
| waiting | grey | `.` |
| skipped / n.a. | dashed; hidden unless "show all" | |

**Where a stage runs:**

| badge | meaning |
|---|---|
| `[L]` | local CPU |
| `[G]` | local 2080 Ti (`gpu.lock`) |
| `[P]` | pod, community cloud |
| `[S]` | pod, secure cloud |
| `[H]` | a person |
| `[U]` | Unity |

**General look and behaviour:**
- Dark theme by default, a light one under `prefers-color-scheme`, system font (Segoe UI), 13 px minimum.
- Works from 1280x720: the DAG scrolls inside its panel, and a list view is always available.
- Keyboard: `g t` takes, `g r` Runpod, `g l` logs, `?` help, `Esc` closes drawers and the topmost dialog only.
  Visible focus rings. A pending `g` chord is resolved before any view shortcut (its second key never reaches the
  view: `g t` is never Video Review's T = tripod). Enter and Space on a focused button, link, player or radio press
  that control, never a view shortcut. A held key's auto-repeat never writes. Every confirmation dialog focuses
  Cancel, so a stray Enter never confirms.

### 6.2 Dashboard and takes list

- **Each take row** has:
  - title ("Leader + Follower x Song"), level / category, variant chip (multi / two / single) and camera count;
  - a **status strip**: one glyph per stage, in DAG order;
  - done count, active run with its progress, remaining ETA p50/p90 and remaining pod cost p50/p90;
  - open gates ("needs you: 2"), last activity.
- **Right column:** a compact Runpod card (6.8), active runs and alerts.

### 6.3 New-capture wizard (`#/new`)

```
New capture   1 Sources  >  2 Videos & camera types  >  3 Dancers  >  4 Dance  >  5 Consent  >  6 Review & create
---------------------------------------------------------------------------------------------------------------------
1 SOURCES
  Videos      [Browse videos...]  or paste a folder: [______________________________] [Scan]
              9 files picked (805 MB)                                   capture type: multi-view (8 used) [auto v]
  Soundtrack  [Browse audio...]  "Ficar Sem Voce.mp3" 3:41  (optional; stays local, never in a repo or public build)
  Take name   [larissa_kadu_2026_11______]  (slug, unique)      Sources: (o) use in place  ( ) copy to dances/<take>/

2 VIDEOS & CAMERA TYPES          (probe runs in the background; suggestions fill in as they arrive)
  cam  file                        res        fps     dur    audio codec  use   mount              zoom       suggestion
  01   01-Demo-Ali.mp4             1080x1920  29.97v  4:12   yes   h264   [x]   (o)Hand ( )Tripod  (o)No ( )Yes  handheld, no zoom 0.91
  04   04-Demo-Tony.mp4            1080x1920  29.97v  4:05   yes   h264   [x]   (o)Hand ( )Tripod  ( )No (o)Yes  handheld, zoom 0.78
  05   05-Demo-Liudmyla.mp4        720x1280   30.00   3:58   yes   hevc*  [x]   ( )Hand ( )Tripod  ( )No ( )Yes  ... probing 40%
  08   08-...-overlay.mp4          1080x1920  29.97v  4:12   yes   h264   [ ]   (excluded: duplicate of 09 with overlays)
  * HEVC: the browser cannot play it -> [Make proxy]      v = variable frame rate (never assumed constant)
  [Accept all suggestions >= 0.70]   Unset types use the general model (handheld + zoom) until you confirm.

3 DANCERS      role    name     stature            shoes                description (for re-ID)            portrait
               lead    [Kadu ]  [5'8"  ] = 1.727 m  (o)sneakers 2.5 cm   [black t-shirt, jeans, shaved head] [Browse]
               follow  [Larissa][165 cm] = 1.650 m  (o)sneakers 2.5 cm   [long red hair, white crop top ...] [Browse]
4 DANCE        level (o)professional ( )intermediate ( )novice   category (o)demo ( )lesson ( )jack_and_jill
               song [______________] artist [______________] date [2026-11-02] venue [________]
               title preview: "Kadu + Larissa x <song>"
5 CONSENT      per dancer: capture  fit_3d  ml_analysis  cloud_processing  name_shown  level_shown  publication  public_repo
               lead        [x]      [x]     [?]          [x]               [?]         [?]          [?]          [ ] (off)
               source rights [__________]   bystanders [__________]
               cloud_processing is required for any pod stage; "?" blocks the stages that need that scope.
6 REVIEW       checklist (green/red) of the intake gates  +  takes/<take>.toml preview (read-only)  +  [Create take]
               -> writes the TOML, starts the probe stage, opens Videos to confirm camera types, then "Plan pipeline".
```

**Stature input** accepts `5'8"`, `173 cm`, `1.727` or `1.727 m` and stores `stature_m`. The shoe-sole presets are
barefoot 0, sneakers 0.025, and custom.

**Camera ids** default to the file's leading two digits, otherwise the pick order. They are editable and must be
unique.

**Capture type.** `auto` gives `single` for 1 used video, `two` for 2, `multi` for 3 or more. The user may force a
variant only downwards (for example, treat 3 videos as two-view). The screen shows what the variant changes, from
the registry's `variants` (for example, single view: monocular HMR + couple refine, scale from statures).

### 6.4 Video Review and camera types (`#/take/<take>/videos`)

```
Videos  larissa_kadu_38    8 used / 9 | camera types: 6 confirmed, 2 suggested | clock reference 06   [Accept suggestions >= 0.70]
+-----------------------------------------------------+  +-----------------------------------------------------+
| 01  01-Demo-Ali.mp4   1080x1920  29.97 VFR  4:12    |  | 05  05-Demo-Liudmyla.mp4  720x1280  30  3:58        |
| [ strip |||||||||||||||||||||||||||||||||||||||| ] |  | [ strip ||||||||||||||||||||||||||||||||||||||| ]  |
| [> play]  0:12.4 / 4:12   audio yes   h264          |  | [Make proxy] HEVC: not playable here                |
| MOUNT  [Handheld]  Tripod        ZOOM  [No]  Yes    |  | MOUNT   Handheld  [Tripod]       ZOOM  [No]  Yes    |
| CONFIRMED by you 2026-10-07 18:02                   |  | SUGGESTED (auto 0.64 - please confirm)  [Confirm]   |
| suggestion: handheld, no zoom  0.91  (camera track) |  | probe_a (before calibration): background parallax  |
|   centre p95 11 cm  axis p95 11 deg  focal x1.00    |  |   0.3 px, jitter 2-12 Hz low, scale drift 0.4 %     |
| pipeline: per-frame 6-dof pose, one focal per video |  | pipeline if tripod: ONE pose for the whole video,   |
|                                                     |  |   ~20 keyframes localised instead of 1151 frames    |
| [Narrate with this camera]  [Exclude]  [Open frames]|  |   (-11 pod-min, est.)  anchors world frame + scale  |
+-----------------------------------------------------+  +-----------------------------------------------------+
+-----------------------------------------------------+
| 07  ...  CONFLICT: you said Tripod, no zoom; the camera track moves 14 cm and zooms x1.12 (probe_b).          |
|     cam_loc / cam_world for 07 are blocked until you choose.   [Keep tripod + no zoom]  [Use handheld + zoom]  |
+---------------------------------------------------------------------------------------------------------------+
```

**Inputs.** Exactly the user's three answers:
- Mount is a two-button segmented control, **Handheld | Tripod** (exclusive).
- Zoom is **No | Yes**, independent of the mount, since a tripod camera can zoom.
- `pans` (tripod only) is a read-only "pans detected" chip from the automatic probe.

**Card states:**
- **unset**: no suggestion and no answer. The pipeline uses handheld + zoom.
- **suggested**: an automatic value, not yet confirmed. Hatched border, with a "Confirm" button.
- **confirmed**: `source = "user"`.
- **conflict**: probe_b contradicts the user's value. Red; the stage gate is open.

**Evidence** comes from the suggestion's `evidence` object:
- probe_a (before calibration): parallax, jitter, zoom scale;
- probe_b (camera tracks): centre p95, axis p95, angular velocity p50, focal ratio, zoom events.

**Effect line.** It comes from the registry's `camera_types.classes`, plus the estimate delta from `progress_eta`:
- tripod: one rigid extrinsic per video, keyframe localisation; anchors the world frame and scale; strong prior for
  sync and SfM;
- tripod + zoom: fixed centre, per-frame rotation and focal;
- handheld: per-frame 6-dof pose, one intrinsic per video;
- handheld + zoom: per-frame pose and focal, log-focal smoothing, zoom-depth guard.

**Saving.** Each click saves at once with `PUT .../camera-type` and an `If-Match` header.
- If the change would make done stages stale, a confirm popover lists them first. Example: "07 -> tripod makes
  cam_loc(07), cam_world, sync, scale_chain, mamma_prepare, mamma_fit, ... stale: re-run p50 1 h 40 min, $2.10 on a
  pod."
- With nothing done yet, no popover appears.

**Keys:**

| key | action |
|---|---|
| `J` / `K` | next / previous video |
| `H` / `T` | handheld / tripod |
| `Z` | toggle zoom |
| `A` | accept suggestion |
| `Enter` | confirm the shown values |
| `Space` | play / pause |
| `N` | narrate with this camera |

At most 2 players run at the same time (3.6). The strips are 12 JPEG frames.

**Narration tool camera panel** (contract for its builder):
- In take mode, its camera picker shows the same chip and popover per camera.
- It uses `GET /api/cameras/types` and `PUT /api/cameras/types/<cam>` on the annotate server, which return and
  accept the `CameraType` shape of 7.4 through `dancecap.camera_types`.
- In `--video` mode (no take) the panel is hidden. The type is set when the video joins a take through the wizard.

### 6.5 Take overview and DAG (`#/take/<take>`)

```
larissa_kadu_38   Kadu + Larissa x <song>   professional / demo   multi-view (8 cams)                 [Narrate moves]
[Overview] [Videos 6/8] [Stages] [Runs] [Analysis]     remaining 3 h 05 (p90 5 h 20)   pods $3.60 (p90 $6.40)
[Plan & run: up to demo_export v] [Dry-run plan] [Resume] [Stop all]                     [Export to Unity] [Open Unity]
 0 INTAKE+CLOCKS    1 FRAMES/AUDIO/2D     2 CAMERAS              3 BODY              4 DETAIL        5 AVATAR ...  7 OUTPUT
 [H] intake    #    [L] ingest      #     [L] cam_map      #     [P] mamma_fit  .    [P] face_pod  . [P] texture .
 [L] probe     #    [G] beats       #   +-[P] cam_loc ------S1-+ (S2, after S1)      ...             ...
 [H] cam_types #    [G] detect      #   | running ####===--- 62%|[L] fit_qa     .
 [L] audio_sync #   [P] dancers_sam #   | 712/1151 fr  ETA 14 (p90 22)|
 ...                [P] ma2d        #   | 3 tripod kf / 5 handheld |
                                        +-------------------------+
 legend: edges = deps (critical path bold), tinted bands = one pod session (S1_front, S2_fit, S_room, S_splats), dashed = optional
```

**Layout:**
- Columns are fixed by the first digit of the registry `phase`: 0 intake and clocks; 1 frames, audio and 2D;
  2 cameras; 3 body; 4 detail; 5 avatar; 6 analysis; 7 output. This gives 8 columns of 180 px.
- Rows are ordered by a barycentre sweep to reduce crossings. Edges are SVG curves.
- Pod sessions are drawn as tinted bands behind their stages, plus a session chip (`S1`) on each node.
- The take's variant decides which stages appear. The **Show other variants** toggle greys in the replaced or added
  stages, for planning.

**Node:** badge, id, status, a progress bar with `done/total unit`, ETA p50 (p90), and one detail line (camera-type
split for camera stages, lanes for MAMMA chunks, the branch for the MAMMA gate: "6 min if the gate passes / 59 min
with the fallback").

**Why the order** (the user's "establish it for future videos"):
- Hovering an edge shows the registry's reason, e.g. "cam_world -> mamma_prepare: cameras are fit inputs; tripod
  views anchor the world frame".
- The **Stages** tab is the list view: topological order, status, where, progress, ETA, cost, last run, gates and
  actions.

**Loops** (`loop_scale`, `loop_refine`, `loop_types`) are drawn as dashed back-edges, with their "when" rule as a
tooltip.

### 6.6 Stage inspector (`#/take/<take>/stage/<id>`, right drawer)

- **Header:** name, id, status with its reason, where it runs, its session group.
- **What it does:** registry name and notes; the CLI, display only.
- **Dependencies:** with their status. Soft dependencies are shown dashed.
- **Outputs:** each `done_when` path with exists, mtime, and "stale because <dep> is newer".
- **Gates:** open or closed, with messages.
- **Estimate:** p50/p90 time and cost, confidence, basis ("measured, n = 6", "estimated from A100 PCIe").
- **Live:** progress steps, lanes and warnings, from the progress events.
- **Last runs:** rc, duration, a tail of `output.log`, and links to review images.
- **Actions:** Run (only) · Run up to here · Resume · Dry-run · Stop · Reveal outputs in Explorer · Open review
  image(s).

### 6.7 Plan and run dialog (the only path to a paid pod)

```
Plan: larissa_kadu_38 -> demo_export (resume)                                       plan 3f2a9c1e  valid 10 min
 stage            where   action         ETA p50 / p90     cost p50 / p90     note
 camera_types     [H]     needs you      -                 -                  2 unconfirmed -> p90 priced as handheld+zoom
 cam_loc          [P] S1  run (stale)    14 / 22 min       (in S1)
 mamma_fit        [P] S2  run            51 / 104 min      (in S2)            gate branch: pass / full-fit fallback
 demo_export      [L]     run            6 / 9 min         -
 POD SESSIONS   (example numbers: the PIPELINE_PROGRESS 8.6 projection of front + MAMMA + avatar chained on one pod)
  S1_front+S2_fit  A100 80GB PCIe community $1.19/h  upload 4.2 GB  181 / 321 min  $3.60 / $6.40  kill switch 340 min  max $6.80
 BUDGET   spent $19.94 + p90 $6.40 = $26.34  <=  cap $30 - reserve $2 = $28.00          OK (green)
 CONSENT  cloud_processing: lead yes, follow yes
 [x] I approve creating 1 paid pod (at most $6.80; it deletes itself after 340 min)
                                                   [Cancel]   [Dry-run bundle (no pod)]   [Run]
```

**When the plan is stale** (it expired, or files changed): `POST runs` returns 412 with the fresh plan. The dialog
redraws and asks again.

**Over budget** (`budget_gate`): the Run button is replaced by the reason. One example: "the p90 would exceed cap -
reserve by $1.20. Confirm camera types (-$0.30), choose another GPU, or ask to raise the cap".

### 6.8 Runpod panel (`#/runpod`) and session view

```
RUNPOD   API ok (checked 18:02:10)   balance $14.06   account spend $0.00/h   progress level L0 (L1 = live GPU + upload)
BUDGET   [##########################|==========.........      |  ]  spent $19.94 | running +$0.00 | plan p50 $3.60 / p90 $6.40
                                    ^ spent     ^ p50   ^ p90   ^ cap - reserve ($28)   ^ cap $30
PODS (1)
 +- dancecap-batch  ...fyk  A100-SXM4-80GB  secure  $1.59/h  ---------------------------------------- connected --+
 |  running 18 min - $0.48 so far - kill switch in 57 min (armed) - GPU 97 %  61/80 GB (L1 only)                   |
 |  session 20261007-065744-batch  avatar/bake pass2 1450/2292 views  ETA 4 min (p90 6)                            |
 |  [upload 9m|boot 1.5m|bake ==========-----|pack|download]                                                      |
 |  [Open session]  [Live probe: off]  [Stop after step (L1)]  [Delete pod...]                                     |
 +-----------------------------------------------------------------------------------------------------------------+
SESSIONS   20261007-065744-batch  ok  29.8 min  $0.79  avatar | 20261007-064623-batch  dry run | ...   [show all 32]
ALERTS     none
```

**Connection pill** (from PIPELINE_PROGRESS 5.5):

| pill | colour |
|---|---|
| `creating`, `starting` | blue |
| `connected` | green |
| `degraded` | amber |
| `lost` | red |
| `ended` | grey |
| `no_key` | grey, "set RUNPOD_API_KEY or create the key file" |
| `api_error` | red, redacted message |

**Red flags:**
- "orphan: pod alive, runner dead";
- "kill switch not armed";
- "untracked spend $x/h";
- "phantom accrual".

**Session view** (`#/runpod/session/<sid>`) renders `progress_eta.snapshot()`:
- a phase Gantt (actual against estimated);
- a step list with progress bars and lanes;
- the kill-switch margin;
- the cost per phase (PIPELINE_PROGRESS 8.5);
- warnings;
- the runner log tail.

### 6.9 Runs and logs

- **Runs:** a table with kind, take, stages, state, started, duration, rc and origin (monitor / cli / external).
  Selecting one shows its log with "follow".
- **Logs:**
  - Sources: monitor server, launcher, any run, any session runner log.
  - Text filter, follow toggle, "copy path".
  - Lines are rendered as text and already redacted by the server.

### 6.10 Narration review, Export to Unity, Open Unity

- **Narrate moves** starts a `tool` run:
  `python -m dancecap.moves annotate --take <take> [--camera C] --no-browser`.
  - When its log prints `[annotate] open http://127.0.0.1:<port>/`, the run's `url` is set, and the monitor calls
    `POST /api/shell/open {tool_run}` to open it as another app window.
  - A second click reuses a running instance for the same take.
  - Before the first moves pass, the panel says it can start "right after clocks and beats" (registry
    `narrate.parallel`).
- **Export to Unity** opens a dialog:
  - fields: capture name (default: the take's last `export/*_demo_export.json`), body fit (default: the take's
    normal motion, `takes/<take>.toml [export] smplx_dir`, else `smplx/`; or `--smplx-dir smplx`, the raw fit;
    `smplx_detail` is refused since the hands steps were retired on 2026-10-07), audio (auto / studio / phone),
    room off;
  - it shows the `unity.lock` owner when the lock is held;
  - it runs `demo_export` and shows the validation summary from `<capture>_demo_export.json`.
- **Open Unity:**
  - The editor is found from `ProjectVersion.txt` and the Hub editor root (a setting).
  - If `head-movement/Temp/UnityLockfile` is locked, it reports "Unity is already open".
  - Otherwise it starts `Unity.exe -projectPath <head-movement>`.
  - Optional **Load capture**: when the Editor is in Play mode, `unity command hm_load --name <capture>
    --project-path <p>` through the Unity CLI.

### 6.11 Analysis and triggers (`#/take/<take>/analysis`, phase 2: shape only)

- **Timeline lanes** on the reference clock:
  - counterbalance windows, floorcraft events, move segments, classifier outputs;
  - each with confidence; low confidence is dashed.
- **Trigger table:** class -> Unity visualisation (for example counterbalance -> CounterbalanceOverlay), with an
  enabled toggle. It reads the `triggers` stage outputs and `taxonomy/triggers.toml`. Saving re-runs `triggers`,
  then offers Export.

## 7. API contract (`dancecap.monitor/1`)

### 7.1 Conventions

- **Transport.** Base `http://127.0.0.1:<port>` (8790-8799, from `server.json`). JSON in UTF-8.
- **Times and units.** Times are epoch seconds (float, UTC); the UI shows local time. Durations are in `_s` seconds,
  money in `_usd` dollars, sizes in `_bytes`.
- **Paths** in responses are workspace-relative with `/` separators, for display. The client never sends a path,
  except the paste-folder field of the picker.
- **Headers:**
  - every write carries `X-Monitor-Token`;
  - GET resources carry an `ETag`, and `If-None-Match` gets a 304;
  - writes to a take TOML need `If-Match: "<toml_etag>"` (`sha1:` of the file bytes).
- **Errors:** `{"error": {"code": "...", "message": "...", "details": {...}}}`.

| HTTP | code(s) |
|---|---|
| 400 | `bad_request` |
| 403 | `forbidden` (host, token, origin) |
| 404 | `not_found` |
| 409 | `conflict` (already running), `budget_gate`, `consent_gate`, `camera_types_gate` |
| 412 | `stale_etag`, `stale_plan` (the body carries the fresh object) |
| 422 | `invalid` (`details.problems[]` = `{field, message}`) |
| 423 | `locked` (`details.lock` = `gpu` or `unity`, `owner`) |
| 429 | `rate_limited` |
| 503 | `runpod_unavailable` |
| 500 | `internal` |

- **Asynchronous actions** return `202 {"run": Run}`. The client follows the run with `GET /api/runs/<id>`.

### 7.2 Endpoints

| method | path | purpose | side effects |
|---|---|---|---|
| GET | `/api/health` | `{app, api, version, instance, pid, started, static_rev}`. Host check only | none |
| GET | `/api/config` | workspace paths (display), python, port, shell, Edge found, Unity project/editor/version, narration port, cap and reserve, `runpod_key: present\|missing`, progress level | none |
| PUT | `/api/settings` | `{shell, theme, idle_exit_min, poll_ms, reserve_usd, unity_editor_root}` -> `work/monitor/settings.json` | file |
| POST | `/api/restart` | restart the backend (3.8) | process |
| POST | `/api/quit` | stop the server; runs continue | process |
| POST | `/api/shell/open` | `{route}` or `{tool_run}`: open an app window on our route or a run's `url`. Never arbitrary URLs | window |
| POST | `/api/shell/reveal` | `{take, what: work\|review\|export\|toml\|run_log, run?}`: open Explorer there | window |
| GET | `/api/state` | dashboard aggregate (7.3 `State`); poll every 2 s with `If-None-Match` | none |
| GET | `/api/stages` | the stage registry, passed through (7.3 `StageDef`) | none |
| GET | `/api/takes` | `[TakeSummary]` | none |
| POST | `/api/takes` | wizard create (7.3 `TakeCreate`). `dry_run: true` returns `{toml_preview, problems, variant}` | writes `takes/<take>.toml`, `work/<take>/intake.json`; starts the probe run |
| GET | `/api/takes/{take}` | `TakeDetail` (with `toml_etag`) | none |
| PATCH | `/api/takes/{take}` | edit `dance`, `dancers.<role>`, `consent` with `If-Match` | TOML (owned keys, 7.4) |
| GET | `/api/takes/{take}/toml` | `{etag, text}`: read-only view | none |
| GET | `/api/takes/{take}/videos` | `[Video]` | none |
| PATCH | `/api/takes/{take}/videos/{cam}` | `{excluded: bool}` with `If-Match` | TOML `exclude` list |
| PUT | `/api/takes/{take}/videos/{cam}/camera-type` | `{mount, zoom}` with `If-Match` -> `{type: CameraType, toml_etag, invalidated: [stage ids]}`. `?preview=1` returns `invalidated` and the cost of re-running without writing | TOML `[cameras."NN"]` |
| POST | `/api/takes/{take}/camera-types/accept` | `{cams: [..] \| "all", min_confidence: 0.7}`: suggestions become `source = "user"` (`by: "accepted"`) | TOML |
| POST | `/api/takes/{take}/camera-types/suggest` | `{cams?}`: re-run the probe for these cameras | `probe` run |
| GET | `/api/takes/{take}/videos/{cam}/stream` | video (Range, 206/416); serves the proxy when one exists | none |
| GET | `/api/takes/{take}/videos/{cam}/thumb?t=12.5&w=320` | JPEG, cached in `work/<take>/monitor/thumbs/` | cache |
| GET | `/api/takes/{take}/videos/{cam}/strip?n=12&h=90` | JPEG contact strip | cache |
| POST | `/api/takes/{take}/videos/{cam}/proxy` | H.264 proxy at `work/<take>/moves/proxy/<cam>.mp4` (shared with the narration tool) | `media` run |
| POST | `/api/fs/pick` | `{kind: videos\|soundtrack\|portrait\|folder}`: native dialog -> `{files: [FileHandle]}` (empty on cancel) | none |
| POST | `/api/fs/scan` | `{folder}` (pasted) -> `{files: [FileHandle]}` of media in that folder, not recursive | none |
| GET | `/api/staging/{handle}` | `FileHandle` with probe and suggestion when ready | none |
| POST | `/api/staging/probe` | `{handles}` -> probe the picked files before a take exists | `probe` run, `work/monitor/staging/` |
| GET | `/api/media/{handle}` · `/thumb` · `/strip` | media of a picked file (Range) | cache |
| GET | `/api/takes/{take}/pipeline` | `PipelineSnapshot` | none |
| GET | `/api/takes/{take}/stages/{stage}` | `StageDetail` | none |
| POST | `/api/takes/{take}/plan` | `PlanRequest` -> `Plan` (no side effects) | none |
| POST | `/api/takes/{take}/runs` | `{plan_hash, approve?: {pods, max_cost_usd}, dry_run?: bool}` -> 202 `Run` (kind `pipeline` or `dry_run`) | processes; pods only with `approve` |
| POST | `/api/takes/{take}/narrate` | `{camera?}` -> 202 `Run` (`tool`), or 200 with the existing one | process |
| POST | `/api/takes/{take}/export` | `{capture, smplx_dir?, audio?, room?, override_lock?}` -> 202 `Run` (`export`) | process |
| GET | `/api/takes/{take}/review` | `[{name, path, mtime, bytes, kind}]` under `work/<take>/review/` | none |
| GET | `/api/takes/{take}/review/{name}` | the image or clip (extension allowlist, no `..`) | none |
| GET | `/api/takes/{take}/analysis` | phase 2: `{lanes: [...], triggers: [...]}` | none |
| PUT | `/api/takes/{take}/triggers` | phase 2: the trigger mapping | `taxonomy/triggers.toml` (local) |
| GET | `/api/runs?take=&active=1&limit=50` | `[Run]` | none |
| GET | `/api/runs/{id}` | `Run` | none |
| GET | `/api/runs/{id}/log?offset=0&max_bytes=65536` | `LogChunk` | none |
| POST | `/api/runs/{id}/stop` | `{mode: after_step\|now}` (only the modes in `run.stop_modes`) | `stop.json` |
| GET | `/api/runpod` | `RunpodSummary` | none (served from cache) |
| POST | `/api/runpod/refresh` | force a pod-list poll (at most 1 per 10 s) | runpodctl call |
| GET | `/api/runpod/gpus` | price table (`remote.gpu_table()`, cached 10 min); loaded only by the plan dialog | runpodctl call |
| POST | `/api/runpod/pods/{pod_id}/delete` | `{confirm: "<last 4 chars of the id>"}`; `dancecap-*` or ledger-known pods only | **deletes a pod** (`remote._delete`, ledger closed) |
| GET | `/api/runpod/orphans` | `{orphans: [{id, name, age_s, cost_per_hr, ledger}]}`: dancecap pods flagged `orphan` now (open ledger row, no active session, no runner process) | none |
| POST | `/api/runpod/cleanup` | `{confirm: "DELETE", pods: [ids]}`: deletes only those ids, each re-checked as an orphan at that moment (409 and nothing deleted otherwise). Never `remote.cleanup_orphans()` (that deletes every `dancecap-*` pod, live ones included). The UI offers it only while an orphan exists and lists the exact ids | deletes orphaned pods |
| GET | `/api/sessions?take=&limit=` | `[SessionSummary]` | none |
| GET | `/api/sessions/{sid}` | `SessionSnapshot` | none |
| GET | `/api/sessions/{sid}/log?offset=` | `LogChunk` of `runner.log` | none |
| POST | `/api/sessions/{sid}/live-probe` | `{enabled}` (PIPELINE_PROGRESS 4.4, read-only probe, L0 sessions only) | ssh probe every 30 s or more |
| POST | `/api/sessions/{sid}/stop` | `{mode: after_step\|now\|delete_pod, confirm?}` (4.3) | runner or pod |
| GET | `/api/budget?plan=<hash>` | `Budget` (projection with that plan) | none |
| GET | `/api/unity` | `{project, editor, version, editor_found, running, lock: {held, owner}, captures: [StreamingAssets folders]}` | none |
| POST | `/api/unity/open` | start the Editor if it is not running | process |
| POST | `/api/unity/load` | `{capture}`: `hm_load` through the Unity CLI, Play mode only | Unity |
| GET | `/api/logs/{source}?offset=` | `source`: `monitor` or `launcher` -> `LogChunk` | none |

### 7.3 Schemas (examples; `//` comments are documentation, not JSON)

**State** (`GET /api/state`):
```json
{"rev": 1842, "now": 1791400000.0,
 "takes": ["TakeSummary ..."],
 "runs_active": ["Run ..."],
 "runpod": {"connection": "ok", "checked": 1791399990.0, "message": null, "pods_running": 0, "sessions_active": 0,
            "progress_level": "L0"},
 "budget": "Budget ...",
 "alerts": [{"id": "camtypes:larissa_kadu_38", "level": "amber", "source": "camera_types",
             "message": "cameras 03, 05 not confirmed - budget p90 +14 pod-min (+$0.30)",
             "link": "#/take/larissa_kadu_38/videos", "since": 1791399000.0}],
 "locks": {"gpu": {"held": false, "owner": null}, "unity": {"held": true, "owner": "<owner.txt first line>"}},
 "monitor": {"api": 1, "version": "1.0.0", "static_rev": "a1b2c3d4", "idle_exit_min": 20}}
```
Alert `level` is one of `red`, `amber` or `info`. Alert `source` is one of `budget`, `pod`, `session`, `stage`,
`camera_types`, `lock`, `consent` or `monitor`.

**TakeSummary:**
```json
{"take": "larissa_kadu_38", "title": "Kadu + Larissa x Ficar Sem Você", "level": "professional", "category": "demo",
 "variant": "multi", "n_videos": 9, "n_used": 8,
 "camera_types": {"confirmed": 6, "suggested": 2, "unset": 0, "conflict": 0, "refused": 0},
 "counts": {"done": 41, "stale": 0, "running": 1, "failed": 0, "blocked": 1, "waiting": 4, "ready": 0, "queued": 0,
            "skipped": 6, "total": 53},
 "strip": [["intake", "done"], ["probe", "done"], ["camera_types", "blocked"], ["...", "..."]],
 "active_runs": ["r-20261007-181502-a1b2"],
 "remaining": {"p50_s": 11100, "p90_s": 19200}, "pod_cost_remaining_usd": {"p50": 3.6, "p90": 6.4},
 "gates_open": 1, "last_activity": 1791399990.0}
```
The `variant` is `multi`, `two` or `single`.

**TakeDetail** = TakeSummary plus:
```json
{"toml_etag": "sha1:9b1c...", "toml_path": "dancecap/takes/larissa_kadu_38.toml",
 "dance": {"level": "professional", "category": "demo", "song": "...", "artist": "...", "date": null, "venue": null},
 "dancers": {"lead": {"name": "Kadu", "stature_m": 1.727, "shoe_sole_m": 0.025, "description": "...", "portrait": false},
             "follow": {"name": "Larissa", "stature_m": 1.651, "shoe_sole_m": 0.025, "description": "...", "portrait": true}},
 "consent": {"lead": {"capture": true, "fit_3d": true, "ml_analysis": "?", "cloud_processing": "?", "...": "..."},
             "follow": {"...": "..."}, "source_rights": "?", "bystanders": "?"},
 "soundtrack": {"name": "Ficar Sem Você.mp3", "present": true},
 "window": [30.0333, 68.4333], "exclude": ["08"],
 "exports": [{"capture": "04_LarissaKaduDemo", "ok": true, "at": 1791399000.0}]}
```
`consent` values are `true`, `false` or `"?"`.

**Video:**
```json
{"cam": "05", "file": "05-Demo-Liudmyla.mp4", "excluded": false,
 "probe": {"duration_s": 238.1, "fps": 30.0, "vfr": false, "width": 720, "height": 1280, "rotation": 0,
           "codec": "hevc", "audio": true, "size_bytes": 61234567},
 "playable": false, "proxy": false,
 "stream": "/api/takes/larissa_kadu_38/videos/05/stream", "strip": "/api/takes/larissa_kadu_38/videos/05/strip?n=12",
 "clock": {"slope": 1.0000123, "offset": -12.31, "reference": false},
 "type": "CameraType ..."}
```
`clock` is `null` before `audio_sync`.

**CameraType.** This one shape is used by the monitor, the narration tool and `dancecap.camera_types`.
```json
{"cam": "05",
 "mount": "tripod",
 "zoom": false,
 "source": "auto",
 "state": "suggested",
 "set_at": 1791399000.0,
 "pans": false, "zoom_events": [],
 "effective": {"mount": "tripod", "zoom": false, "class": "tripod"},
 "suggestion": {"mount": "tripod", "zoom": false, "pans": false, "confidence": 0.64, "by": "probe_a", "at": 1791398000.0,
                "evidence": {"parallax_px": 0.3, "jitter_hf": 0.05, "zoom_scale": 1.004,
                             "centre_p95_m": null, "axis_p95_deg": null, "angvel_p50_dps": null, "focal_ratio": null,
                             "zoom_events": []}},
 "conflict": null,
 "effects": ["one pose for the whole video (keyframe localisation)", "one intrinsic set",
             "a constant camera track for sync and the dancers"],
 "model_check": null,
 "estimate_delta": {"pod_min_p50": -11, "usd_p50": -0.22, "basis": "unit count (no tripod measured yet)"}}
```

| field | values and meaning |
|---|---|
| `mount` | `"handheld"`, `"tripod"`, or `null` when unset |
| `zoom` | `true`, `false`, or `null` |
| `source` | `"user"` (confirmed, incl. accepted suggestions), `"auto"` (suggestion only) or `"default"` (nothing known) |
| `state` | `"unset"`, `"suggested"`, `"confirmed"` or `"conflict"` |
| `effective` | what the pipeline uses; the default is handheld + zoom |
| `effective.class` | `tripod`, `tripod_zoom`, `handheld` or `handheld_zoom` (registry `camera_types.classes`) |
| `pans`, `zoom_events` | automatic, read-only (TOML keys of the same name); `zoom_events` = `[[t0, t1, focal_ratio], ...]` in reference seconds |
| `suggestion.by` | `probe_a` (pre-calibration) or `probe_b` (camera tracks) |
| `conflict` | `null`, or `{field, user, measured, message}` |
| `model_check` | `null` before a cameras run, else what it did with the type (report.json `cams.<cam>.camera_type`): `{declared, model, passed, why, at, current, refused}`; `refused` = the stage could not use the CURRENT declared type and kept a safer model (shown as a warning, summary count `refused`, an amber alert; never a block) |
| `effective` (policy) | the `user` policy (confirmed answers only); every cameras run records the policy it used and the stale check compares under that policy (`pipeline/camtypes.effective_class`) |

**Camera-type rules:**
- The user always wins. A data conflict never overrides the user's value: it sets `state: conflict` and opens the
  `camera_types` gate for that camera.
- `accept` copies the suggestion into `mount` and `zoom` with `source: "user"`.
- Evidence the camera track cannot judge (an attached camera: centre and focal fixed by the attach model:
  `evidence.mount_judgeable` / `zoom_judgeable` false) is never a conflict; probe A decides those fields.
- A measured 'tripod' below confidence 0.7 pre-fills handheld (`evidence.mount_lean: "tripod"`).
- In the inputs, picking one field of a pre-filled suggestion does not save it: Confirm (or picking both fields)
  does; a confirmed answer changes field by field. The zoom question reads "Zoomed during the video".
- `cam_loc` estimate factors are the implemented ones (`camtypes.LOC_FACTOR`): locked tripod 0.1 (p90 1.1), every
  other class 1.0.

**FileHandle** (picker and staging):
```json
{"handle": "h-6f1c2a9b0d3e4f51", "kind": "video", "name": "05-Demo-Liudmyla.mp4",
 "path": "C:/.../dances/Larissa-Kadu-demo/05-Demo-Liudmyla.mp4", "size_bytes": 61234567,
 "probe": null, "suggestion": null, "probe_state": "queued",
 "media": "/api/media/h-6f1c2a9b0d3e4f51", "strip": "/api/media/h-6f1c2a9b0d3e4f51/strip?n=8"}
```
- `path` is for display only; the picker returned it to this user.
- `probe_state` is `queued`, `running`, `done` or `failed`.
- Handles live for one server instance. `work/monitor/staging/<handle>.json` keeps the probe results for a restart.

**TakeCreate** (`POST /api/takes`):
```json
{"dry_run": false, "take": "larissa_kadu_2026_11", "source_mode": "reference",
 "videos": [{"handle": "h-...", "cam": "01", "excluded": false, "type": {"mount": "handheld", "zoom": false}}],
 "soundtrack": {"handle": "h-..."},
 "capture_type": "auto",
 "dancers": {"lead": {"name": "...", "stature_m": 1.727, "shoe_sole_m": 0.025, "description": "...", "portrait": {"handle": "h-..."}},
             "follow": {"...": "..."}},
 "dance": {"level": "professional", "category": "demo", "song": "...", "artist": "...", "date": "2026-11-02", "venue": ""},
 "consent": {"lead": {"capture": true, "fit_3d": true, "ml_analysis": "?", "cloud_processing": true, "name_shown": "?",
                      "level_shown": "?", "publication": "?", "public_repo": false},
             "follow": {"...": "..."}, "source_rights": "...", "bystanders": "..."}}
```
- `source_mode` is `"reference"` (use in place) or `"copy"` (into `dances/<take>/`, local).
- `soundtrack` may be `null`, and so may each dancer's `portrait`.
- `capture_type` is `"auto"`, `"multi"`, `"two"` or `"single"`.
- A video `type` may be omitted, in which case the probe suggestion applies when it lands.

The response is `201 {"take", "toml_path", "toml_etag", "problems": [], "run": Run}` (the probe run), or
`422 {"error": {"code": "invalid", "details": {"problems": [{"field": "dancers.follow.stature_m", "message": "..."}]}}}`.

**StageDef** (`/api/stages`). The final registry is passed through. The UI relies on these fields:
- `id, name, phase, runs_on, deps, soft_deps, variants, optional, session, gates, outputs, done_when, cli` (display
  only), `progress`, `est`, `exists`, `parallel`, `notes`;
- optional `helps: [{stage, why}]`, which feeds the "why this order" tooltips;
- optional `alt_runs_on`, which shows "Run here instead";
- optional `heavy`.

**PipelineSnapshot:**
```json
{"take": "larissa_kadu_38", "variant": "multi", "rev": 1842,
 "columns": ["0 intake+clocks", "1 frames/audio/2D", "2 cameras", "3 body", "4 detail", "5 avatar", "6 analysis", "7 output"],
 "stages": [{
   "id": "cam_loc", "column": 2, "runs_on": "pod", "session": "S1_front", "optional": false, "in_variant": true,
   "status": "running", "status_reason": "front session 20261007-...: loc:02 712/1151",
   "progress": {"done": 712, "total": 1151, "unit": "frames", "frac": 0.62, "step": "front/loc:02", "lanes": 3},
   "eta": {"p10_s": 600, "p50_s": 840, "p90_s": 1320, "basis": "live + history (n=6)", "confidence": "measured"},
   "cost": {"so_far_usd": 0.41, "p50_usd": 0.28, "p90_usd": 0.44},
   "camera_types": {"tripod": ["05"], "tripod_zoom": [], "handheld": ["01", "02", "03"], "handheld_zoom": ["04", "07", "09"]},
   "last_run": {"run": "r-...", "rc": 0, "ended": 1791300000.0, "seconds": 1004},
   "outputs": {"fresh": false, "stale_because": ["camera_types (05 changed)"]},
   "gates": [{"id": "camera_types", "open": false}],
   "review": ["review/cameras_loc_02.jpg"],
   "actions": ["run", "run_upto", "dry_run", "stop", "reveal"]}],
 "edges": [["cam_map", "cam_loc", "dep"], ["camera_types", "cam_loc", "dep"], ["fit_qa", "mamma_prepare", "loop_scale"]],
 "groups": [{"id": "S1_front", "stages": ["dancers_sam", "ma2d", "cam_loc", "cam_world", "sync", "scale_chain"], "sid": "20261007-..."}],
 "critical_path": ["camera_types", "cam_loc", "cam_world", "sync", "mamma_prepare", "mamma_fit", "fit_qa", "texture", "demo_export"],
 "branches": [{"stage": "mamma_fit", "label": "seam gate", "options": [{"name": "pass", "eta_p50_s": 360}, {"name": "fallback", "eta_p50_s": 3540}]}],
 "remaining": {"p10_s": 9000, "p50_s": 11100, "p90_s": 19200}, "pod_cost_remaining_usd": {"p50": 3.6, "p90": 6.4},
 "gates_open": [{"id": "camera_types", "stage": "camera_types", "message": "2 cameras unconfirmed", "link": "#/take/larissa_kadu_38/videos"}],
 "next_actions": [{"label": "Confirm camera types", "route": "#/take/larissa_kadu_38/videos"},
                  {"label": "Plan & run up to demo_export", "plan": {"target": "demo_export", "scope": "upto", "mode": "resume"}}]}
```

**Status and edges:**
- Stage `status` is one of `done`, `stale`, `ready`, `waiting`, `blocked`, `queued`, `running`, `failed`, `stopped`,
  `skipped`.
- Precedence: running > failed > blocked > stale > done > ready > waiting > skipped. The rules come from the
  registry `status_rule`.
- Edge kinds are `dep`, `soft` and `loop_*`.

**StageDetail** = the stage entry above plus:
- `def` (StageDef);
- `outputs_detail: [{path, exists, mtime, bytes, stale_because}]`;
- `deps_detail: [{id, status}]`;
- `runs: [Run]` (last 10);
- `log_tail: [lines]`;
- `progress_steps: [...]` (from PIPELINE_PROGRESS `snapshot().steps`);
- `warnings: [...]`.

**PlanRequest and Plan:**
```json
{"target": "demo_export", "scope": "upto", "mode": "resume", "include_optional": [],
 "options": {"gpus": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"], "clouds": ["COMMUNITY"], "max_cost_usd": null}}
```
- `scope` is `"only"` (this stage), `"upto"` (it and its not-done ancestors) or `"all"`.
- `mode` is `"resume"` (skip fresh outputs) or `"force"`.

```json
{"plan_hash": "sha1:3f2a9c1e...", "take": "larissa_kadu_38", "variant": "multi", "created": 1791400000.0, "valid_for_s": 600,
 "steps": [{"stage": "camera_types", "action": "blocked", "runs_on": "human", "reason": "2 cameras unconfirmed"},
           {"stage": "cam_loc", "action": "run", "runs_on": "pod", "session": "S1_front", "reason": "stale: camera type 05 changed",
            "est": {"p50_s": 840, "p90_s": 1320}, "argv_display": ["python", "-m", "dancecap.front_pod", "..."]}],
 "sessions": [{"id": "S1_front+S2_fit", "jobs": ["front", "mamma", "avatar"], "gpu": "NVIDIA A100 80GB PCIe", "cloud": "COMMUNITY", "price_per_hr": 1.19,
               "upload_bytes": 4191000000, "est_min": {"p50": 181, "p90": 321}, "cost_usd": {"p50": 3.6, "p90": 6.4},
               "max_minutes": 340, "max_cost_usd": 6.8}],
 "totals": {"wall_s": {"p10": 9000, "p50": 11100, "p90": 19200}, "local_s": 900, "pod_cost_usd": {"p50": 3.6, "p90": 6.4}},
 "budget": {"spent": 19.94, "after_p50": 23.54, "after_p90": 26.34, "cap": 30.0, "reserve": 2.0, "gate": "ok"},
 "consent": {"cloud_processing": {"lead": true, "follow": true}, "ok": true},
 "camera_type_band": {"unconfirmed": ["03", "05"], "p90_extra_pod_min": 14, "p90_extra_usd": 0.30},
 "warnings": ["MAMMA seam gate: p90 includes the full-fit fallback (P(fail) prior 0.5)"]}
```
- Step `action` is one of `run`, `skip_done`, `rerun_stale`, `blocked` or `n/a`.
- Budget `gate` is `ok`, `over_reserve` or `over_cap`.
- **The hash** is computed over steps, sessions, prices and the input files' mtimes.
- **`POST runs` revalidation.** The server recomputes the plan.
  - A changed hash gives 412 `stale_plan` with the new plan.
  - `approve.pods` must equal `len(sessions)`, and `approve.max_cost_usd` must be at least the sum of the sessions'
    `max_cost_usd`. Otherwise the server returns 409.
  - The gate must not be `over_cap`, and when it is `over_reserve` the request must also carry
    `approve.over_reserve: true`.

**Run:**
```json
{"id": "r-20261007-181502-a1b2", "kind": "pipeline", "take": "larissa_kadu_38", "stages": ["cam_loc", "..."],
 "plan_hash": "sha1:3f2a...", "origin": "monitor", "state": "running", "pid": 34792,
 "created": 1791400502.1, "started": 1791400502.9, "ended": null, "rc": null,
 "current": {"stage": "cam_loc", "session_sid": "20261007-181530-front-larissa_kadu_38", "step": "front/loc:02"},
 "progress": {"frac": 0.31, "eta": {"p50_s": 9800, "p90_s": 17000}},
 "url": null, "stop_modes": ["after_step", "now"],
 "log": "/api/runs/r-20261007-181502-a1b2/log", "argv_display": ["python", "-m", "dancecap.pipeline", "run", "..."]}
```
- `origin` is `monitor`, `cli` or `external`.
- `state` is one of `starting`, `running`, `waiting_human`, `stopping`, `done`, `failed`, `stopped` or `lost`.
- `url` is set for `tool` runs.

**RunpodSummary:**
```json
{"connection": "ok", "checked": 1791400000.0, "message": null,
 "account": {"balance_usd": 14.06, "spend_per_hr_usd": 0.0, "spend_limit": null},
 "pods": [{"id": "podidexample04", "name": "dancecap-batch", "managed": true, "gpu": "NVIDIA A100-SXM4-80GB", "cloud": "SECURE",
           "cost_per_hr": 1.59, "status": "running", "runtime_status": "running", "connection": "connected",
           "created": 1791392238.2, "age_s": 1080, "cost_so_far_usd": 0.48,
           "kill": {"at": 1791396738.0, "source": "estimate", "armed": true, "margin_s": 3400},
           "gpu_util": null, "session": {"sid": "20261007-065744-batch", "take": "larissa_kadu_38", "phase": "avatar/bake/pass2"},
           "ledger": "open", "flags": []}],
 "sessions_active": ["20261007-065744-batch"], "progress_level": "L0", "budget": "Budget ..."}
```

| field | values |
|---|---|
| `connection` | `ok`, `no_key`, `api_error`, `unknown` |
| pod `connection` | `creating`, `starting`, `connected`, `degraded`, `lost`, `ended` |
| `kill.source` | `probe` or `estimate` |
| `gpu_util` | `{util_pct, mem_used_mib, mem_total_mib, at}` with L1 or the live probe |
| `ledger` | `open`, `closed`, `missing` |
| `flags` | `orphan`, `killswitch_unarmed`, `untracked`, `phantom_accrual` |

**Budget** (PIPELINE_PROGRESS 8.1-8.2):
```json
{"cap": 30.0, "reserve": 2.0, "spent": 19.94, "remaining": 10.06, "open_accrual_usd": 0.0,
 "running": {"p50_usd": 0.0, "p90_usd": 0.0, "worst_usd": 0.0}, "queued": {"p50_usd": 0.0, "p90_usd": 0.0},
 "projected": {"t50": 19.94, "t90": 19.94, "worst": 19.94}, "level": "green", "messages": [],
 "untracked_spend_per_hr_usd": 0.0, "corrected_spent": null}
```
`level` is `green`, `amber` or `red`.

**SessionSummary and SessionSnapshot.**
- `SessionSummary`: `{sid, take, label, origin, state, started, ended, pod_id, gpu, cost_usd, jobs, progress_level}`.
- `SessionSnapshot` = `progress_eta.snapshot(session_dir)` (PIPELINE_PROGRESS 10.1:
  `{phases[], steps[], eta {p10, p50, p90}, kill_margin, cost {so_far, p50, p90, worst}, warnings[]}`), plus
  `{sid, take, origin, runner_alive, progress_level, pod: Pod | null, stop_modes}`.

**LogChunk:**
```json
{"offset": 0, "next": 18233, "size": 18233, "eof": true, "lines": ["[pod-runner 10:27:04] avatar: ok (1097s)"],
 "truncated_head": false}
```
- At most `max_bytes` per call. Lines are cut only at `\n`.
- Lines are redacted and are plain text.

### 7.4 Camera-type storage and TOML writes (`dancecap.camera_types`, shared)

The keys are exactly those of `docs/PIPELINE.md` section 4.1, plus `set_at`:

```toml
[cameras."05"]              # camera type (Pipeline Monitor / narration tool; block written by dancecap.camera_types)
mount = "tripod"            # handheld | tripod                 (user)
zoom = false                # true | false, independent of mount (user)
source = "user"             # user = confirmed | auto = suggestion, not confirmed yet
set_at = "2026-10-07T18:02:11Z"
# written by the probe, kept for the record; never overrides the user:
suggested = { mount = "tripod", zoom = false, pans = false, confidence = 0.64, by = "probe_a", at = "2026-10-07T17:40:00Z" }
pans = false                # tripod only, automatic
zoom_events = []            # automatic: [t0, t1, focal ratio] in reference seconds
```

`take.camera_type(cam)`, the reader that PIPELINE.md 4.1 proposes for stage code, delegates to
`dancecap.camera_types.effective()`, so every reader and writer goes through one module.

**Module API** (used by the monitor and the narration server):
- `load(take) -> (list[CameraType], toml_etag)`
- `set_type(take, cam, mount, zoom, if_match) -> (CameraType, toml_etag)`
- `accept(take, cams, min_confidence, if_match)`
- `write_suggestion(take, cam, suggestion)`
- `effective(take, cam) -> class`

**Reading.** Per-camera subtables are taken only for keys matching `^[0-9]{2}$`. `scale_rule`, `scale_reference` and
`loc` stay reserved `[cameras]` keys. Existing takes have no per-camera table, so they read as `unset`, and their
effective class is `handheld_zoom`, which is today's behaviour.

**Owned-block writer.** No tomlkit dependency.
- The `[cameras."NN"]` blocks are **owned by the module**. A block runs from its header to the next header or the
  end of the file. It is rewritten as a whole, or appended at the end of the file.
  - TOML allows a `[cameras."NN"]` header after other tables, even though `[cameras]` appears earlier.
  - The writer emits the fixed comments shown above. A comment a person types inside an owned block is not kept;
    every other comment in the file is.
  - This is the "line editor" option of PIPELINE.md 4.1 (never `tomllib` plus a dump).
- Every byte outside the owned blocks is preserved exactly.
- Wizard-only keys (`[dance]`, `[dancers.*]`, `[consent]`) are written when the take is created. Later edits through
  `PATCH /api/takes` rewrite only `key = value` lines of known scalar keys inside those tables. A key missing from
  the file is inserted at the end of its table.
- **Steps of a write:**
  1. Check `If-Match` against `sha1` of the current bytes. A mismatch is a 412.
  2. Build the new text.
  3. `tomllib.loads` it, and compare the parsed tables with the intended values.
  4. Back up the old file to `work/<take>/monitor/toml_backups/<ts>.toml`. This location is git-ignored; `takes/`
     itself is tracked.
  5. Write a temp file, then `os.replace` it, with retries on `PermissionError`.
- **Concurrency.** The monitor, the narration server and a human editor can all write. The etag catches the first
  two. A human editor holding unsaved changes can still overwrite: the UI warns when the file's mtime changes under
  an open form.

**Wizard TOML** (intake contract, aligned with the registry's `intake` stage; the backend may extend it):
- `name`, `source_dir` (relative to `takes/`), `alignment = "../work/<take>/clock/alignment.json"` (written later by
  `audio_sync`) and `exclude`.
- `[cameras."NN"] file = "<name in source_dir>"` plus the camera-type keys.
- `[dance]`, `[dancers.<role>]` and `[consent]`.
- `window` is written later by the `window` stage.

**Loading.** The monitor loads takes with its own tolerant reader (tomllib, like `DanceMeta`), never `Take.load`,
which requires `alignment` and `window`.

### 7.5 Refresh model

**Polling cadence:**
- `GET /api/state`: every 2 s while the window is visible, every 10 s while hidden (`document.visibilityState`).
- The visible detail endpoint (`pipeline`, `videos`, `sessions/<sid>`, a run) is polled at the same cadence. Logs
  are polled every 1 s while "follow" is on.

**Change detection.** `rev` is a server-side counter. It rises when a stat-watched file changes (take TOMLs,
`done_when` outputs of visible takes, session dirs, ledger, run dirs), when a runpodctl poll result changes, or when
a run changes state. Responses carry `ETag: "rev-<n>"`.

**Server-side pollers** (one thread each, all read-only):
- local files: a `stat` every 2 s, reading only on a change;
- `runpodctl pod list`: every 30 s while a session is active or the Runpod view is open, otherwise every 5 min;
- `runpodctl user` (through `remote.status()`): every 5 min.

**SSE** (`GET /api/events`, `event: rev`) is optional later. waitress flushes streamed chunks (its `send_bytes`
default is 1), but each stream holds one of the 16 threads. Polling stays the contract.

### 7.6 Frontend structure (no build step)

```
static/index.html                 shell + <meta name="dancecap-token">, loads app.js as a module
static/api.js                     fetch wrapper: token header, ETag cache, error envelope -> exceptions, 412/409 handling
static/app.js                     router (hash), store (State), poller, alerts bar, status bar
static/views/{dashboard,wizard,videos,take,dag,stage,plan,runpod,session,runs,logs,settings,analysis}.js
static/components/{cameraType,progress,gauge,strip,dialog,table,logview,player}.js
static/style.css  icons.svg  manifest.webmanifest
```

**Rendering rules:**
- DOM builders with `textContent`. Every server string is untrusted text: no `innerHTML` with data, and no `eval`.
- Plain functions and one store: no framework.
- SVG for the DAG, the gauges and the Gantt bars.

## 8. Build plan and acceptance (for the builder stages; create no pods)

| step | deliverable | acceptance |
|---|---|---|
| 1 | launcher, `monitor --launch/--serve`, `/api/health`, static shell, single instance, logs | Double-click shows a window in 3 s or less. A second double-click starts **no** second server (one `monitor.lock` holder, checked with psutil) and opens a window. Closing every window: the server exits after `idle_exit_min`. `build.ps1` succeeds under PS 5.1. |
| 2 | read-only views: `/api/state`, takes, pipeline snapshot (registry + `done_when` mtimes), Runpod panel on the ledger, sessions with L0 parsing, logs | Shows both existing takes and the 32 sessions. "No pods running" matches `runpodctl pod list`. **No console window appears during 10 min of polling** (regression: under the server, a runpodctl or ffprobe child reports `GetConsoleWindow() == 0`). |
| 3 | `dancecap.camera_types`, Video Review, narration-tool camera panel | Owned-block round trip on copies of both take TOMLs: bytes are identical outside the blocks, `tomllib` parses, reserved keys are untouched, a concurrent write gives 412. Suggestions from the registry's `validation_larissa_kadu_38` values. |
| 4 | wizard: picker, staging probe, TOML preview, create | A dry-run on the larissa_kadu sources gives a TOML that the tolerant reader loads. Statures parse (`5'8"` -> 1.727). An HEVC file offers a proxy. |
| 5 | runs: `runwrap`, plan, local stage runs, stop, narration, export, Unity | `stop now` on a fake long run kills the child and runs `finally`. The narration tool opens in an app window and its microphone prompt appears. Export refuses with 423 while `work/unity.lock/` exists. |
| 6 | pod launches through plan + approve | **Tested with `dry_run` and a fake `remote` only.** 412 on a stale plan, 409 on the budget gate (fake ledger with open rows), and a consent gate. |
| 7 | `head-movement/Assets/Editor/PipelineMonitorMenu.cs`, `Tools/PipelineMonitor.cmd` | Unity compiles them. The menu opens the monitor. `git grep` in head-movement finds no absolute path, user name, take or dancer name. |

**API tests** use the Flask test client, without a server:
- every route's status codes;
- Host, token and Origin enforcement;
- path traversal on `review/` and media;
- the pod delete endpoint refuses non-`dancecap-*` pods (fake `_ctl`);
- log redaction keeps sha256 hex and masks `rpa_...`.

**UI smoke test.** The Claude Browser pane at `http://127.0.0.1:8790/` (`read_page`) loads every route without
console errors.

## 9. Risks and open questions

- **Stop "now" for pod runners** depends on interrupting the main thread plus killing the ssh child, so that
  `pod_session`'s `finally` deletes the pod. Until that is verified on a fake pod, the UI offers only
  `delete_pod` for sessions (4.3).
- **Defender ML on a new unsigned exe.** The risk is low for a 6 KB csc build without a packed payload, and the exe
  has no Mark-of-the-Web, so SmartScreen does not prompt. The `.lnk` and `.cmd` paths remain fallbacks. A
  false-positive report can go to Microsoft if it ever happens.
- **The Edge app window carries Edge's taskbar icon.** `manifest.webmanifest` plus an optional "Install as app" gives
  it its own icon. Edge policy changes to `--app` are unlikely but possible, so `shell: browser` stays as the
  fallback.
- **The microphone grant is per origin.** The narration tool should keep port 8765. Its `free_port` fallback leads
  to a re-prompt.
- **No tripod has been measured yet.** All 8 larissa_kadu phones are handheld, so the tripod saving shown in the UI
  is an estimate from unit counts, and the UI labels it "est.".
- **Camera-type thresholds are uncalibrated.** A suggestion below 0.7 confidence keeps the gate open.
- **Owned-block TOML editing drops comments inside `[cameras."NN"]` blocks.** This is documented, and the block's
  first line says so.
- **The workspace `.gitignore` needs `!/tools/`** and an ignore line for `tools/PipelineMonitor/PipelineMonitor.exe`.
  That is the builder's edit, not this stage's.
- **Shared ownership of the take TOML.** The registry/backend stage (dag) also writes take TOML keys (`window`,
  `alignment`). All writers should use the owned-block writer of 7.4.
- **History is thin.** The ETA and cost bands are wide, and the UI shows the basis ("estimated from 1 run").
- **Runs of other workflows** (the larissa_kadu_38 re-bake, the demo workflow) appear as `external`. They are never
  restarted, and the only action on them is `delete_pod` with a typed confirmation.

Sources (checked 2026-10-07):
- [pywebview `edgechromium.py`: WebView2 through pythonnet/WinForms; no PermissionRequested handler](https://github.com/r0x0r/pywebview/blob/master/webview/platforms/edgechromium.py)
- [pywebview issue #909: getUserMedia in pywebview](https://github.com/r0x0r/pywebview/issues/909)
- [pywebview 6.0 release notes](https://pywebview.flowrl.com/blog/pywebview6)
- [PyInstaller issue #5848: Trojan:Win32/Wacatac.B!ml after compiling](https://github.com/pyinstaller/pyinstaller/issues/5848)
- [PyInstaller discussion #5877: antivirus flags the exe](https://github.com/orgs/pyinstaller/discussions/5877)
- [waitress arguments: `send_bytes` defaults to 1 and streamed app_iter chunks are flushed](https://docs.pylonsproject.org/projects/waitress/en/stable/arguments.html)
- [Edge command line: `--app`, `--user-data-dir` (Microsoft Q&A)](https://learn.microsoft.com/en-us/answers/questions/1162281/what-is-syntax-for-command-line-to-run-edge-in-kio)
- [Edge UserDataDir policy](https://learn.microsoft.com/en-us/deployedge/microsoft-edge-policies/userdatadir)
- [VBScript deprecation timeline (Microsoft 365 Developer Blog)](https://devblogs.microsoft.com/microsoft365dev/how-to-prepare-vba-projects-for-vbscript-deprecation/)
- Local measurements in section 1 (scratchpad prototypes: `PipelineMonitor.cs` 6 KB build, the `launcher.json`
  parse, venv launcher pid, console-window inheritance, import timing).

---

## Addendum: integration stage (2026-10-07, after the end-to-end proof)

User guide: `docs/PIPELINE_MONITOR_GUIDE.md`. API and UI changes made while integrating (all additive):

| Change | Where |
|---|---|
| `POST /api/takes/<take>/export-dry-run` {capture, smplx_dir?, audio?, room?} -> 202 {run (kind `dry_run`, stages [demo_export]), dry_run: true}. Runs `python -m dancecap.pipeline export <take> --capture <Name> --dry-run`: writes nothing; lists the inputs found / missing, the folders it would replace, the Unity lock and the triggers it would carry; rc 0 when the required inputs exist. Its own route on purpose: an older server answers 404, never a real export. The Export dialog has a **Dry run** button. | server.py, pipeline/export.py, actions.js |
| Plan request `options.assume_reviews` (bool): a **what-if** plan, dry-run only. Human steps get the action `pause` ("a real run pauses here for you"), review gates do not block their descendants, a stage blocked by a gate you can clear (consent, camera types) is planned "what-if: runs once you act". Plan fields `what_if`, and `halts_at` lists the pauses. `POST /runs` refuses a real run of a what-if plan (422); so does `python -m dancecap.pipeline run`. CLI: `plan --what-if`, `run --what-if`. UI: **What-if…** in the take header and a toggle in the plan dialog (Run disabled). | plan.py, orchestrator.py, __main__.py, actions.js, takehead.js |
| Plan field `camera_types`: [{cam, state, source, mount, zoom, class, suggested, effects}] (the effective class each camera stage applies), and `steps[].camera_types` {class: [cams]} on the camera steps. The plan dialog shows "Camera types this plan applies" and per-step chips. | plan.py, actions.js |
| One pod per job: a pod session whose job cannot be narrowed (S1_front) is `blocked` ("waits with the S1_front pod job for <stage> …") while one of its job's stages is blocked (e.g. dancers_sam after labels_review). Before, a new take's plan created an S1 pod for cam_loc alone while the job would also attempt SAM before the labels review. | plan.py |
| Soft deps that run in the same run are waited for by the orchestrator (any result) and counted in the plan's wall time; a soft dep that is also a descendant is ignored (no wait cycle). | orchestrator.py, plan.py |
| Plan dialog: **Unity capture name** field (pre-filled with the take's last capture, else the next free `NN_<TakeCamelCase>`), so `demo_export` can be planned for a new take. | actions.js |
| Stage drawer: buttons for an open review gate (fit_qa: Looks good / Reject; geoveto: Apply the veto / Keep as measured / Reject) -> `POST /gates/<stage>`. `waiting_human` runs are final everywhere (no live progress bar, no Stop). | stagedrawer.js, runs.js |
| `runpod.progress_level` in `/api/state` = the best level of the running sessions (was always L0, while the session view said L1). On a test tree (DANCECAP_PIPELINE_ROOT) the account's real pods are not reported as untracked spend. | runpod_view.py |
| Probe A is not measured twice: the wizard's staged suggestion carries the file signature; intake writes it to `work/<take>/probe/camera_types_suggested.json`, and the probe stage reuses it (4 synthetic 34 s videos: 55 s -> 1.7 s). | camtypes.py, intake.py |
| Failing pod jobs: the session snapshot carries `task_failures` {failed, skipped, first, tasks} parsed from the runner log (front_pod's `<task>: failed` / `skipped (dependency failed ...)` lines) and a warning; the Runpod summary lists `sessions_failing` and `/api/state` raises an amber alert for a RUNNING session with failed tasks (its pod bills until the job ends). A job status `failed: <reason>` (what remote_jobs writes) now makes the session `failed` (was shown `ok`), and an ended session no longer shows its last log line as 'Now'. Found on a live larissa_kadu_full front session whose ingest failed on the pod (89 failed, 151 skipped, $0.46). | runpod_view.py, server.py |
| Plan budget: `budget.cap` follows `remote.cap()` on the real tree ($80 since 2026-10-07; the plan dialog showed $30 while the gate used $80). | estimate.py |
| Trigger rules: the viewer's new `balance` layer (DanceLayers.Handles) with the element `couple.balance_axis` on counterbalance and pivot triggers. | taxonomy/triggers.toml, pipeline/triggers.py |

---

## 11. Redesign around the farm (2026-10-08)

User directive: "I want all the old jobs cleared out of the monitor. It's confusing the way it's designed right now. I want to
see the equivalent of a hard drive defragmentation graphic ... how much of the timeline is fully completed, partially completed,
or not completed at all." The contract is `docs/FARM.md` section 7; this section is the implementation.

- **Front page = the farm view** (`#/` and `#/farm/<dance>`), nothing else in the main navigation but Older tools and Settings.
  The takes dashboard, take pages, new-capture wizard, camera-type review, runs, Runpod sessions and logs are unchanged but only
  reachable through `#/legacy` ("Older tools"); the shell polls `/api/state` only while one of them is open.
- **Old jobs cleared, evidence kept**: `python -m dancecap.pipeline archive-legacy` moves `work/pipeline/runs/*`,
  `plans/*`, `runs_archive/*` and each take's `pipeline/events.jsonl` to `work/pipeline/archive/<date>/` (with a MANIFEST.json); a run
  that is still alive is skipped. `work/runpod/sessions/*` and the ledger are never touched; the farm view reads only the newest
  session of each fit segment.
- **Backend** (all additive; `monitor/server.py` got one registration call): `pipeline/farm_view.py` (the bundle),
  `pipeline/fit_sessions.py` (read-only adapter of the live `dancecap.fit_full` sessions: segment, pod, GPU, price, age, step, job
  states fit / refine / quality, ETA from `progress_eta`, heartbeat from `progress/puller.json`, cost), `pipeline/farm_actions.py`
  (plan -> approve -> detached run; every command line is built here, the client never sends an argv),
  `pipeline/monitor/farm_routes.py` (the REST routes), `pipeline/legacy_archive.py`, `pipeline/farm_demo.py`.
- **API** (same security model as the rest: Host allowlist, cookie on GET, token + JSON + Origin on writes):
  `GET /api/farm?dance=` (bundle, ETag), `GET /api/farm/log`, `POST /api/farm/fill/plan`, `/fill/run`, `/pull`,
  `/watcher/restart`, `/pods/stop`, `/coverage/refresh`, `/map-png`, `/reveal`.
- **Bundle** `dancecap.monitor.farm/1`: `coverage` (cells `{i, t0, t1, k0, k1, s, g, n, m, f, r[], c, o[], live}`, counts, seconds,
  percent), `chunks`, `jobs`, `queue`, `pods`, `budget`, `volume`, `world`, `watcher`, `events`, `alerts`, `outputs`, `actions`.
  It reads `work/<dance>/farm/{coverage,state}.json` (core's schemas), `farm_volume.json`, `farm/watcher.pid`, and for the live
  fits `work/<dance>/fit/{runner_<seg>,commitments}.json` and `work/runpod/sessions/*-fit-<seg>/`.
- **Frontend**: `static/views/farm.js`, `static/components/defrag.js` (canvas painter shared by the display and the PNG export),
  `static/farm.css`; `app.js` routes `#/` to the farm view and lazy-polls `/api/state` for the older tools.
- **Launcher**: `tools/PipelineMonitor/PipelineMonitor.exe [--view farm] [--dance D]` (rebuilt with `build.ps1`; the exe is only a
  trampoline), `head-movement/Tools/PipelineMonitor.cmd` and the Unity menu item are unchanged.
- Tests: `tests/test_farm_monitor.py` (data layer, adapter, routes, actions' safety rules, archive, demo).
