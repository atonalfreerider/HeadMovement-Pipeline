# Offline test speech for the narration tool's smoke test: Windows SAPI text-to-speech into a 16 kHz mono wav.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File sapi_synth.ps1 -Out test.wav -Script script.txt
#
# script.txt: one item per line, either "pause <seconds>" or text to speak. -Voice picks an installed voice
# (default: the first enabled en-US voice). Prints JSON {voice, items:[{text|pause, ...}]}.
param(
  [Parameter(Mandatory = $true)][string]$Out,
  [Parameter(Mandatory = $true)][string]$Script,
  [string]$Voice = "",
  [int]$Rate = 0
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
if ($Voice) { $s.SelectVoice($Voice) }
else {
  $v = @($s.GetInstalledVoices() | Where-Object { $_.Enabled -and $_.VoiceInfo.Culture.Name -eq "en-US" })
  if ($v.Count -gt 0) { $s.SelectVoice($v[0].VoiceInfo.Name) }
}
$s.Rate = $Rate
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$s.SetOutputToWaveFile($Out, $fmt)
$p = New-Object System.Speech.Synthesis.PromptBuilder
foreach ($line in (Get-Content -Encoding UTF8 $Script)) {
  $l = $line.Trim()
  if ($l -eq "") { continue }
  if ($l -match '^pause\s+([0-9.]+)$') { $p.AppendBreak([TimeSpan]::FromSeconds([double]$Matches[1])) }
  else { $p.AppendText($l) }
}
$s.Speak($p)
$name = $s.Voice.Name
$s.Dispose()
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
ConvertTo-Json -InputObject @{ voice = $name; out = $Out } -Compress
