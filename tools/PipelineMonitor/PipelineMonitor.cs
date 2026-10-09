// Pipeline Monitor launcher (docs/PIPELINE_MONITOR.md 3.2). A tiny trampoline with no logic of its own:
//   1. one launch at a time (named mutex), so a double double-click starts one monitor;
//   2. reads launcher.json next to this exe (paths relative to the exe, forward slashes);
//   3. starts "<python> -m dancecap.pipeline monitor --launch [--take T] [--view V] [--dance D] [--capture C]" with no console
//      window; that command reuses a running server (single instance) or starts one, then opens the Edge app window;
//   4. waits up to start_timeout_s and reports a failure with the tails of launcher.log and monitor.log (the server's
//      own reason, e.g. "another monitor holds work/monitor/monitor.lock"). Every message box is shown AFTER the
//      mutex is released, so a further double-click while a message is open is not swallowed.
// Everything else lives in Python, so updating the monitor never needs a rebuild of this exe.
// C# 5 only (the in-box .NET Framework 4.8 csc): no $"", no ?., no expression-bodied members.
// Build: powershell -NoProfile -ExecutionPolicy Bypass -File build.ps1
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Web.Script.Serialization;
using System.Windows.Forms;

static class PipelineMonitorLauncher
{
    const string Title = "Pipeline Monitor";

    static string Str(Dictionary<string, object> cfg, string key, string dflt)
    {
        object v;
        return cfg.TryGetValue(key, out v) && v is string && ((string)v).Length > 0 ? (string)v : dflt;
    }

    static int Int(Dictionary<string, object> cfg, string key, int dflt)
    {
        object v;
        if (!cfg.TryGetValue(key, out v) || v == null) return dflt;
        try { return Convert.ToInt32(v); } catch (Exception) { return dflt; }
    }

    static string Quote(string a)
    {
        if (a.Length > 0 && a.IndexOfAny(new[] { ' ', '\t', '"' }) < 0) return a;
        var sb = new StringBuilder("\"");
        int slashes = 0;
        foreach (char c in a)
        {
            if (c == '\\') { slashes++; continue; }
            if (c == '"') { sb.Append('\\', slashes * 2 + 1); sb.Append('"'); slashes = 0; continue; }
            sb.Append('\\', slashes); slashes = 0; sb.Append(c);
        }
        sb.Append('\\', slashes * 2);
        sb.Append('"');
        return sb.ToString();
    }

    static string Tail(string path, int lines)
    {
        try
        {
            if (!File.Exists(path)) return "(no log yet)";
            string[] all;
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
            using (var sr = new StreamReader(fs, Encoding.UTF8))
                all = sr.ReadToEnd().Split('\n');
            int from = Math.Max(0, all.Length - lines);
            return string.Join("\n", all, from, all.Length - from).Trim();
        }
        catch (Exception e) { return "(cannot read the log: " + e.Message + ")"; }
    }

    static void Log(string path, string msg)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            File.AppendAllText(path, DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss") + " [exe] " + msg + "\n", new UTF8Encoding(false));
        }
        catch (Exception) { }
    }

    // The message to show once the launch mutex is released (null: nothing to show).
    static string pendingMsg;
    static MessageBoxIcon pendingIcon = MessageBoxIcon.Error;

    static void Fail(string msg)
    {
        pendingMsg = msg;
        pendingIcon = MessageBoxIcon.Error;
    }

    static void Info(string msg)
    {
        pendingMsg = msg;
        pendingIcon = MessageBoxIcon.Information;
    }

    // Only these options are forwarded; Python validates the values again.
    static readonly Dictionary<string, Regex> Allowed = new Dictionary<string, Regex>
    {
        { "--take", new Regex("^[a-z0-9][a-z0-9_]{0,47}$") },
        { "--view", new Regex("^(videos|overview|runpod|new|farm)$") },
        { "--dance", new Regex("^[a-z0-9][a-z0-9_]{0,47}$") },
        { "--capture", new Regex("^[A-Za-z0-9_-]{1,64}$") },
    };

    [STAThread]
    static int Main(string[] argv)
    {
        bool created;
        int rc;
        using (var mutex = new Mutex(true, @"Local\dancecap-pipeline-monitor-launch", out created))
        {
            bool owned = created;
            if (!owned)
            {
                try { owned = mutex.WaitOne(15000); }
                catch (AbandonedMutexException) { owned = true; }
                if (!owned) return 0; // another launch is still starting the monitor
            }
            try { rc = Launch(argv); }
            finally { mutex.ReleaseMutex(); }
        }
        if (pendingMsg != null) MessageBox.Show(pendingMsg, Title, MessageBoxButtons.OK, pendingIcon);
        return rc;
    }

    static int Launch(string[] argv)
    {
        string here = AppDomain.CurrentDomain.BaseDirectory;
        string cfgPath = Path.Combine(here, "launcher.json");
        Dictionary<string, object> cfg;
        try
        {
            string text = File.Exists(cfgPath) ? File.ReadAllText(cfgPath, Encoding.UTF8) : "{}";
            cfg = new JavaScriptSerializer().Deserialize<Dictionary<string, object>>(text) ?? new Dictionary<string, object>();
        }
        catch (Exception e)
        {
            Fail("launcher.json is not valid JSON (use forward slashes in paths):\n" + cfgPath + "\n\n" + e.Message);
            return 2;
        }
        string python = Path.GetFullPath(Path.Combine(here, Str(cfg, "python", "../../.venv-nopo4d/Scripts/python.exe")));
        string cwd = Path.GetFullPath(Path.Combine(here, Str(cfg, "cwd", "../../dancecap")));
        string log = Path.GetFullPath(Path.Combine(here, Str(cfg, "log", "../../dancecap/work/monitor/logs/launcher.log")));
        int timeout = Math.Max(5, Int(cfg, "start_timeout_s", 30));
        if (!File.Exists(python))
        {
            Fail("Python was not found:\n" + python + "\n\nSet \"python\" in\n" + cfgPath);
            return 2;
        }
        if (!Directory.Exists(cwd))
        {
            Fail("The dancecap folder was not found:\n" + cwd + "\n\nSet \"cwd\" in\n" + cfgPath);
            return 2;
        }

        var args = new List<string>();
        object a;
        if (cfg.TryGetValue("args", out a) && a is ArrayList)
            foreach (object x in (ArrayList)a) args.Add(Convert.ToString(x));
        if (args.Count == 0) args.AddRange(new[] { "-m", "dancecap.pipeline", "monitor", "--launch" });
        for (int i = 0; i + 1 < argv.Length; i += 2)
        {
            Regex rx;
            if (Allowed.TryGetValue(argv[i], out rx) && rx.IsMatch(argv[i + 1])) { args.Add(argv[i]); args.Add(argv[i + 1]); }
        }
        var line = new StringBuilder();
        foreach (string s in args) { if (line.Length > 0) line.Append(' '); line.Append(Quote(s)); }

        var psi = new ProcessStartInfo(python, line.ToString());
        psi.WorkingDirectory = cwd;
        psi.UseShellExecute = false;
        psi.CreateNoWindow = true;
        psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
        psi.EnvironmentVariables["PYTHONUTF8"] = "1";
        Log(log, "start " + python + " " + line);
        Process p;
        try { p = Process.Start(psi); }
        catch (Exception e)
        {
            Fail("Could not start Python:\n" + python + "\n\n" + e.Message);
            return 3;
        }
        if (!p.WaitForExit(timeout * 1000))
        {
            Log(log, "still starting after " + timeout + " s");
            Info("The monitor is slow to start (still working after " + timeout + " s).\nIt opens by itself when ready; see\n" + log);
            return 0;
        }
        Log(log, "rc " + p.ExitCode);
        if (p.ExitCode != 0)
        {
            string mlog = Path.Combine(Path.GetDirectoryName(log), "monitor.log");
            Fail("The Pipeline Monitor did not start (exit code " + p.ExitCode + ").\n\nLast lines of " + log + ":\n\n" + Tail(log, 8)
                 + "\n\nLast lines of the server log " + mlog + ":\n\n" + Tail(mlog, 12)
                 + "\n\nIf it says another monitor holds monitor.lock, a hung monitor is still running: end the python.exe"
                 + " whose pid is in work/monitor/server.json (Task Manager > Details), then double-click again.");
            return p.ExitCode;
        }
        return 0;
    }
}
