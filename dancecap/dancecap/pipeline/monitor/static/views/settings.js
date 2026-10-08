// Settings (#/settings): window shell, theme, idle exit, polling, budget reserve, Unity editor folder; backend
// restart / quit; configuration facts (never the Runpod key itself: only "present" or "missing").
import { h, replace, prefs } from "../util.js";
import { get, put, post } from "../api.js";
import { toast, errorText, confirmDialog } from "../components/dialog.js";
import { pill } from "../components/status.js";

export default {
  title: () => "Settings",
  async mount(root, route, app) {
    let cfg;
    try { cfg = await get("/api/config"); } catch (e) { replace(root, h("p", { class: "crit-text" }, errorText(e))); return {}; }
    const s = cfg.settings || {};
    const shell = h("select", null, [["edge-app", "Edge app window (default)"], ["browser", "default browser tab"], ["pywebview", "pywebview (not installed)"]]
      .map(([v, l]) => h("option", { value: v, selected: (s.shell || "edge-app") === v }, l)));
    const theme = h("select", null, [["system", "follow Windows"], ["dark", "dark"], ["light", "light"]]
      .map(([v, l]) => h("option", { value: v, selected: (s.theme || "system") === v }, l)));
    const idle = h("input", { type: "number", min: "0", max: "1440", value: s.idle_exit_min ?? 20, style: { width: "100px" } });
    const poll = h("input", { type: "number", min: "1000", max: "10000", step: "500", value: s.poll_ms ?? 2000, style: { width: "100px" } });
    const reserve = h("input", { type: "number", min: "0", max: "100", step: "0.5", value: s.reserve_usd ?? cfg.reserve ?? 2, style: { width: "100px" } });
    const unityRoot = h("input", { type: "text", value: s.unity_editor_root || "", placeholder: "C:\\Program Files\\Unity\\Hub\\Editor", style: { width: "100%" } });
    theme.addEventListener("change", () => {
      if (theme.value === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = theme.value;
      prefs.set("theme", theme.value === "system" ? null : theme.value);
    });
    async function saveSettings() {
      const body = { shell: shell.value, theme: theme.value, idle_exit_min: Number(idle.value), poll_ms: Number(poll.value), reserve_usd: Number(reserve.value) };
      if (unityRoot.value.trim()) body.unity_editor_root = unityRoot.value.trim();
      try { const r = await put("/api/settings", body); app.config.settings = r; toast("Settings saved.", "good", { timeout: 2500 }); }
      catch (e) { toast(errorText(e), "crit"); }
    }
    const u = cfg.unity || {};
    replace(root,
      h("div", { class: "page-head" }, h("h1", null, "Settings")),
      h("div", { class: "grid2" },
        h("div", { class: "card" }, h("h3", { class: "section-title" }, "Monitor"),
          h("div", { class: "form-grid" },
            h("label", { class: "field" }, h("span", null, "Window"), shell),
            h("label", { class: "field" }, h("span", null, "Theme"), theme),
            h("label", { class: "field" }, h("span", null, "Exit when idle (min, 0 = never)"), idle),
            h("label", { class: "field" }, h("span", null, "Refresh every (ms)"), poll),
            h("label", { class: "field" }, h("span", null, "Budget reserve ($)"), reserve)),
          h("label", { class: "field", style: { marginTop: "12px" } }, h("span", null, "Unity Hub editor folder"), unityRoot),
          h("div", { class: "row", style: { marginTop: "14px" } }, h("button", { class: "btn primary", onclick: saveSettings }, "Save"),
            h("span", { class: "spacer" }),
            h("button", { class: "btn", title: "Reload the backend code (runs keep going)", onclick: async () => {
              try { await post("/api/restart", {}); toast("Restarting the backend… the page reconnects by itself.", "info"); } catch (e) { toast(errorText(e), "crit"); }
            } }, "Restart backend"),
            h("button", { class: "btn danger", onclick: async () => {
              if (!await confirmDialog("Quit the monitor server?", "Runs and pod sessions keep going; double-click Pipeline Monitor to come back.", { okLabel: "Quit", danger: true })) return;
              try { await post("/api/quit", {}); toast("The server is stopping. You can close this window.", "info", { timeout: 0 }); } catch (e) { toast(errorText(e), "crit"); }
            } }, "Quit server"))),
        h("div", { class: "card" }, h("h3", { class: "section-title" }, "This installation"),
          h("div", { class: "kv" },
            h("span", null, "workspace"), h("span", null, cfg.workspace || "–"),
            h("span", null, "package root"), h("span", { class: "mono small" }, cfg.root || "–"),
            h("span", null, "python"), h("span", null, cfg.python || "–"),
            h("span", null, "port"), h("span", null, String(cfg.port || location.port)),
            h("span", null, "Edge"), h("span", null, cfg.edge_found ? "found" : "not found (default browser is used)"),
            h("span", null, "Runpod API key"), h("span", null, cfg.runpod_key === "present" ? pill("present", "good") : pill("missing", "warn", "set RUNPOD_API_KEY or create the key file")),
            h("span", null, "budget"), h("span", null, `cap $${cfg.cap ?? "?"} · reserve $${cfg.reserve ?? "?"}`),
            h("span", null, "Unity"), h("span", null, `${u.version || "?"} ${u.editor_found ? "(editor found)" : "(editor not found)"}`),
            h("span", null, "Unity project"), h("span", { class: "mono small" }, u.project || "–"),
            h("span", null, "narration tool port"), h("span", null, String(cfg.narration_port || 8765)),
            h("span", null, "progress level"), h("span", null, cfg.progress_level || "L0"),
            h("span", null, "compute policy"), h("span", { class: "small" }, cfg.compute_policy || "")),
          h("h3", { class: "section-title", style: { marginTop: "16px" } }, "Launch"),
          h("p", { class: "small" }, "Double-click ", h("code", null, "tools\\PipelineMonitor\\PipelineMonitor.exe"), " (or the Desktop shortcut, or ",
            h("code", null, "head-movement\\Tools\\PipelineMonitor.cmd"), ", or Unity's menu ", h("strong", null, "Head Movement › Pipeline Monitor"),
            "). A second launch reuses this server and only opens another window."),
          h("p", { class: "small muted" }, "Press ? anywhere for keyboard shortcuts."))));
    return {};
  },
};
