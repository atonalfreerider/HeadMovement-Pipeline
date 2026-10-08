# Builds PipelineMonitor.exe with the C# compiler that ships with .NET Framework 4.8 (no admin, no SDK, no PyInstaller).
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\PipelineMonitor\build.ps1
# ASCII only and Windows PowerShell 5.1-safe (docs/PIPELINE_MONITOR.md 3.6).
$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
if (-not (Test-Path $csc)) { $csc = Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe' }
if (-not (Test-Path $csc)) { Write-Error 'csc.exe (.NET Framework 4.x) was not found'; exit 1 }
$src = Join-Path $here 'PipelineMonitor.cs'
$out = Join-Path $here 'PipelineMonitor.exe'
$ico = Join-Path $here 'monitor.ico'
$cscArgs = @('-nologo', '-target:winexe', '-optimize+', ('-out:' + $out),
             '-r:System.Windows.Forms.dll', '-r:System.Web.Extensions.dll')
if (Test-Path $ico) { $cscArgs += ('-win32icon:' + $ico) }
$cscArgs += $src
& $csc @cscArgs
if ($LASTEXITCODE -ne 0) { Write-Error ('csc failed with exit code ' + $LASTEXITCODE); exit $LASTEXITCODE }
Write-Host ('built ' + $out)
