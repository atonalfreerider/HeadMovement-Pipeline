# Install the hm-guard git hooks into this clone (local git config only).  Same options as install_hooks.sh:
#   powershell -File tools/install_hooks.ps1 [--no-test | --check | --uninstall]
# The work is done by install_hooks.sh under Git Bash, because the hooks themselves are sh scripts.
$git = (Get-Command git -ErrorAction Stop).Source
$bash = Join-Path (Split-Path (Split-Path $git -Parent) -Parent) "bin\bash.exe"
if (-not (Test-Path $bash)) { throw "Git Bash not found next to git ($bash)" }
$root = (& git rev-parse --show-toplevel).Trim()
Set-Location $root
& $bash "tools/install_hooks.sh" @args
exit $LASTEXITCODE
