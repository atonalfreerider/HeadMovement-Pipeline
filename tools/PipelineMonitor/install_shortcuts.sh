#!/usr/bin/env bash
# Linux shortcuts of the Pipeline Monitor (the Windows ones: install_shortcuts.ps1): an application-menu entry and a
# Desktop launcher, both running PipelineMonitor.sh. Run once:  bash tools/PipelineMonitor/install_shortcuts.sh
set -eu
HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
chmod +x "$HERE/PipelineMonitor.sh"
ICON="$HERE/monitor.png"
if [ ! -f "$ICON" ] && command -v ffmpeg >/dev/null; then
  ffmpeg -v error -y -i "$HERE/monitor.ico" "$ICON" || true
fi
entry() {
  cat <<EOF
[Desktop Entry]
Type=Application
Name=Pipeline Monitor
Comment=Head Movement capture pipeline: farm, analyses, Runpod sessions
Exec="$HERE/PipelineMonitor.sh"
Icon=$ICON
Terminal=false
Categories=Development;
EOF
}
APPS="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
mkdir -p "$APPS"
entry > "$APPS/pipeline-monitor.desktop"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESK" ]; then
  entry > "$DESK/pipeline-monitor.desktop"
  chmod +x "$DESK/pipeline-monitor.desktop"
  # GNOME only runs a desktop file the user trusts
  gio set "$DESK/pipeline-monitor.desktop" metadata::trusted true 2>/dev/null || true
fi
echo "installed: $APPS/pipeline-monitor.desktop${DESK:+ and $DESK/pipeline-monitor.desktop}"
