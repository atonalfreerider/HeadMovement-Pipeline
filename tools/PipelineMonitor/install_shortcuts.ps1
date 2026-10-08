# Creates "Pipeline Monitor.lnk" on the Desktop and in Start menu > Programs > Head Movement, pointing at the exe.
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\PipelineMonitor\install_shortcuts.ps1
# Per-user, no admin. ASCII only and Windows PowerShell 5.1-safe.
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $here 'PipelineMonitor.exe'
if (-not (Test-Path $exe)) {
  Write-Error ('Build the launcher first: powershell -NoProfile -ExecutionPolicy Bypass -File "' + (Join-Path $here 'build.ps1') + '"')
  exit 1
}
$shell = New-Object -ComObject WScript.Shell
$dirs = @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('Programs')) 'Head Movement'))
foreach ($dir in $dirs) {
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $path = Join-Path $dir 'Pipeline Monitor.lnk'
  $lnk = $shell.CreateShortcut($path)
  $lnk.TargetPath = $exe
  $lnk.WorkingDirectory = $here
  $lnk.IconLocation = ($exe + ',0')
  $lnk.Description = 'dancecap Pipeline Monitor: pipeline stages, Runpod progress, ETA and budget (local)'
  $lnk.Save()
  Write-Host ('shortcut: ' + $path)
}
