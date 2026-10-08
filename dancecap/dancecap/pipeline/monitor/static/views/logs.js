// Logs (#/logs): the monitor server, the launcher, any run's output.log, any session's runner.log. Lines arrive
// redacted by the server (keys and tokens masked) and are rendered as text.
import { h, replace, prefs } from "../util.js";
import { get, listOf } from "../api.js";
import { logView } from "../components/logview.js";
import { toast } from "../components/dialog.js";

export default {
  title: () => "Logs",
  async mount(root, route, app) {
    const sources = [
      { id: "monitor", label: "monitor server", url: "/api/logs/monitor", path: "dancecap/work/monitor/logs/monitor.log" },
      { id: "launcher", label: "launcher", url: "/api/logs/launcher", path: "dancecap/work/monitor/logs/launcher.log" },
    ];
    try {
      for (const r of listOf(await get("/api/runs?limit=40"), "runs")) {
        sources.push({ id: "run:" + r.id, label: `run ${r.id} · ${r.kind} ${r.take || ""} ${(r.stages || []).join(",")}`, url: `/api/runs/${r.id}/log`,
          path: `dancecap/work/pipeline/runs/${r.id}/output.log` });
      }
    } catch { /* optional */ }
    try {
      for (const s of listOf(await get("/api/sessions?limit=40"), "sessions")) {
        sources.push({ id: "session:" + s.sid, label: `session ${s.sid} · ${(s.jobs || []).join(",")} (${s.state})`, url: `/api/sessions/${s.sid}/log`,
          path: `dancecap/work/runpod/sessions/${s.sid}/runner.log` });
      }
    } catch { /* optional */ }
    let cur = sources.find((s) => s.id === prefs.get("log_source", "monitor")) || sources[0];
    const sel = h("select", { "aria-label": "Log source", style: { minWidth: "420px" } },
      h("optgroup", { label: "monitor" }, sources.filter((s) => !s.id.includes(":")).map((s) => h("option", { value: s.id, selected: s === cur }, s.label))),
      h("optgroup", { label: "runs" }, sources.filter((s) => s.id.startsWith("run:")).map((s) => h("option", { value: s.id, selected: s === cur }, s.label))),
      h("optgroup", { label: "Runpod sessions (runner.log)" }, sources.filter((s) => s.id.startsWith("session:")).map((s) => h("option", { value: s.id, selected: s === cur }, s.label))));
    const pathEl = h("code", null, cur.path);
    const lv = logView({ url: cur.url, height: "calc(100vh - 260px)" });
    sel.addEventListener("change", () => {
      cur = sources.find((s) => s.id === sel.value) || sources[0];
      prefs.set("log_source", cur.id);
      pathEl.textContent = cur.path;
      lv.setSource(cur.url);
    });
    replace(root,
      h("div", { class: "page-head" }, h("h1", null, "Logs"), sel,
        h("span", { class: "muted small" }, pathEl),
        h("button", { class: "btn small", onclick: async () => {
          try { await navigator.clipboard.writeText(cur.path); toast("Path copied.", "good", { timeout: 2000 }); }
          catch { toast(cur.path, "info"); }
        } }, "Copy path")),
      lv.el);
    return { destroy() { lv.destroy(); } };
  },
};
