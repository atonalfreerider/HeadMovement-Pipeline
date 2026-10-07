# Offline speech recognition with the Windows desktop recogniser (System.Speech / SAPI, en-US), for the dancecap
# narration tool (dancecap/moves/annotate/asr.py engine 'sapi'). Nothing leaves the machine.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File sapi_transcribe.ps1 -Wav narration.wav -PhraseFile phrases.txt
#
# Two passes over the same 16 kHz mono PCM wav:
#   grammar    the move vocabulary (one phrase per line in -PhraseFile) between optional wildcards, so a move name
#              inside other talk is found; recognised AND rejected results are reported (a rejected result is the
#              recogniser's best guess with a low confidence)
#   dictation  free English dictation (talk, unknown move names)
# Output (stdout, UTF-8 JSON): {"grammar":[result...], "dictation":[result...], "recognizer": "..."}
#   result = {kind: ok|rejected, grammar, text, confidence, start, duration, words:[{word, start, end, prob}]}
#   start / duration in seconds of the input; word start / end RELATIVE to the result's start ("..." = wildcard).
param(
  [Parameter(Mandatory = $true)][string]$Wav,
  [string]$PhraseFile = "",
  [string]$Culture = "en-US",
  [switch]$NoDictation,
  [switch]$Check
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$src = @"
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Speech.Recognition;
using System.Threading;
public class DcSapiWord { public string word; public double start; public double end; public double prob; }
public class DcSapiResult {
  public string kind; public string grammar; public string text; public double confidence;
  public double start; public double duration; public List<DcSapiWord> words = new List<DcSapiWord>();
}
public static class DancecapSapi {
  static DcSapiResult Convert(RecognitionResult r, string kind) {
    var res = new DcSapiResult();
    res.kind = kind;
    res.grammar = r.Grammar != null ? r.Grammar.Name : "";
    res.text = r.Text; res.confidence = r.Confidence;
    res.start = r.Audio != null ? r.Audio.AudioPosition.TotalSeconds : -1;
    res.duration = r.Audio != null ? r.Audio.Duration.TotalSeconds : 0;
    for (int i = 0; i < r.Words.Count; i++) {
      var w = r.Words[i];
      var sw = new DcSapiWord(); sw.word = w.Text; sw.prob = w.Confidence; sw.start = -1; sw.end = -1;
      try {
        var wa = r.GetAudioForWordRange(w, w);
        if (wa != null) { sw.start = wa.AudioPosition.TotalSeconds; sw.end = sw.start + wa.Duration.TotalSeconds; }
      } catch (Exception) {}
      res.words.Add(sw);
    }
    return res;
  }
  public static List<DcSapiResult> Run(string wav, string[] phrases, string culture, bool dictation) {
    var ci = new CultureInfo(culture);
    var output = new List<DcSapiResult>();
    using (var eng = new SpeechRecognitionEngine(ci)) {
      if (dictation) {
        var dict = new DictationGrammar(); dict.Name = "dictation";
        eng.LoadGrammar(dict);
      } else {
        var gb = new GrammarBuilder(); gb.Culture = ci;
        var lead = new GrammarBuilder(); lead.AppendWildcard();
        gb.Append(lead, 0, 1);
        gb.Append(new Choices(phrases));
        var tail = new GrammarBuilder(); tail.AppendWildcard();
        gb.Append(tail, 0, 1);
        var g = new Grammar(gb); g.Name = "moves";
        eng.LoadGrammar(g);
      }
      eng.SetInputToWaveFile(wav);
      eng.BabbleTimeout = TimeSpan.Zero; eng.InitialSilenceTimeout = TimeSpan.Zero;
      eng.EndSilenceTimeout = TimeSpan.FromMilliseconds(400);
      var done = new ManualResetEvent(false);
      eng.SpeechRecognized += (s, e) => { lock (output) { output.Add(Convert(e.Result, "ok")); } };
      eng.SpeechRecognitionRejected += (s, e) => {
        if (e.Result != null && e.Result.Words.Count > 0) { lock (output) { output.Add(Convert(e.Result, "rejected")); } }
      };
      eng.RecognizeCompleted += (s, e) => done.Set();
      eng.RecognizeAsync(RecognizeMode.Multiple);
      done.WaitOne();
    }
    return output;
  }
}
"@
Add-Type -TypeDefinition $src -ReferencedAssemblies System.Speech
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$recs = @([System.Speech.Recognition.SpeechRecognitionEngine]::InstalledRecognizers() | Where-Object { $_.Culture.Name -eq $Culture })
if ($recs.Count -eq 0) { throw "no $Culture speech recognizer installed" }
if ($Check) { ConvertTo-Json -InputObject @{ recognizer = $recs[0].Description; culture = $Culture } -Compress; exit 0 }
$phr = @()
if ($PhraseFile) { $phr = @(Get-Content -Encoding UTF8 $PhraseFile | Where-Object { $_.Trim() -ne "" }) }
$grammar = @()
if ($phr.Count -gt 0) { $grammar = @([DancecapSapi]::Run($Wav, [string[]]$phr, $Culture, $false)) }
$dictation = @()
if (-not $NoDictation) { $dictation = @([DancecapSapi]::Run($Wav, [string[]]@(), $Culture, $true)) }
$out = @{ recognizer = $recs[0].Description; grammar = $grammar; dictation = $dictation }
ConvertTo-Json -InputObject $out -Depth 6 -Compress
