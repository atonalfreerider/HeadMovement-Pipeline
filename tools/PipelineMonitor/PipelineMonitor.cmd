@echo off
rem Fallback launcher when PipelineMonitor.exe is not built: starts the dancecap Pipeline Monitor
rem from the workspace venv (a console window shows for a few seconds; the exe avoids it).
setlocal
set "HERE=%~dp0"
set "PY=%HERE%..\..\.venv-nopo4d\Scripts\python.exe"
if not exist "%PY%" (
  echo Python not found: "%PY%"
  pause
  exit /b 1
)
pushd "%HERE%..\..\dancecap"
"%PY%" -m dancecap.pipeline monitor --launch %*
set RC=%ERRORLEVEL%
popd
if not "%RC%"=="0" (
  echo The monitor did not start ^(exit %RC%^). See dancecap\work\monitor\logs\launcher.log
  pause
)
exit /b %RC%
