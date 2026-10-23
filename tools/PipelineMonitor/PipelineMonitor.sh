#!/usr/bin/env bash
# Linux launcher of the dancecap Pipeline Monitor (the Windows one is PipelineMonitor.exe / .cmd beside this file).
# Starts the local server (127.0.0.1:8790..8799) if it is not running and opens a Chrome / Edge --app window on it.
# Extra arguments are passed through: --take T, --view V, --dance D, --replace, --no-window.
set -u
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY=""
for cand in "$ROOT/.venv/bin/python" "$ROOT/.venv-nopo4d/bin/python"; do
  if [ -x "$cand" ]; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "Python venv not found under $ROOT (.venv or .venv-nopo4d)" >&2
  exit 1
fi
# runpodctl is installed per user (~/.local/bin); a desktop launcher does not always have it on PATH
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT/dancecap" || exit 1
"$PY" -m dancecap.pipeline monitor --launch "$@"
RC=$?
if [ "$RC" != "0" ]; then
  echo "The monitor did not start (exit $RC). See dancecap/work/monitor/logs/launcher.log" >&2
fi
exit $RC
