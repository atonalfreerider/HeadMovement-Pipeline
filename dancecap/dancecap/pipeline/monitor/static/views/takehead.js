// Shared take header: title, metrics, the action bar (Plan & run, Dry-run, Stop, Narrate, Export, Open Unity) and
// the tabs (Pipeline, Stages, Videos & camera types, Runs, Analysis & triggers).
import { h, fmtEta, fmtCostBand } from "../util.js";
import { pill } from "../components/status.js";
import { camTypeSummary } from "./dashboard.js";

const VARIANT = { multi: "multi-view", two: "two-view", single: "single-view" };

export function takeHeader(app, take, detail, tab) {
  const t = detail || app.takeSummary(take) || { take };
  const c = t.counts || {};
  const runs = (app.state && app.state.runs_active || []).filter((r) => r.take === take);
  const pipelineRun = runs.find((r) => r.kind === "pipeline" || r.kind === "dry_run");
  const A = async () => app.actions();
  const sound = t.soundtrack ? (t.soundtrack.present ? `♪ ${t.soundtrack.name}` : `♪ ${t.soundtrack.name} (missing)`) : "no soundtrack";
  const tabs = [
    ["dag", `#/take/${take}`, "Pipeline"],
    ["stages", `#/take/${take}/stages`, "Stages"],
    ["videos", `#/take/${take}/videos`, "Videos & camera types"],
    ["runs", `#/take/${take}/runs`, "Runs"],
    ["analysis", `#/take/${take}/analysis`, "Analysis & triggers"],
  ];
  const ct = t.camera_types || {};
  const needCt = (ct.unset || 0) + (ct.suggested || 0) + (ct.conflict || 0);
  return h("div", { class: "take-header" },
    h("div", { class: "take-head" },
      h("div", { class: "th-title" },
        h("h1", null, t.title || take),
        h("div", { class: "th-sub row" },
          h("span", { class: "chip mono" }, take),
          [t.level, t.category].filter(Boolean).length ? h("span", null, [t.level, t.category].filter(Boolean).join(" / ")) : null,
          h("span", null, `${VARIANT[t.variant] || t.variant || "?"} (${t.n_used ?? "?"} of ${t.n_videos ?? "?"} videos)`),
          h("span", { class: "muted" }, sound),
          t.dancers ? h("span", { class: "muted" }, Object.entries(t.dancers).map(([r, d]) =>
            `${r} ${d.name || "?"} ${d.stature_m ? d.stature_m.toFixed(3) + " m" : ""}`).join(" · ")) : null)),
      h("div", { class: "metrics" },
        h("div", { class: "metric" }, h("div", { class: "m-label" }, "Stages done"),
          h("div", { class: "m-value" }, `${c.done ?? 0}/${c.total ?? 0}`),
          h("div", { class: "m-sub" }, [c.stale ? `${c.stale} stale` : null, c.failed ? `${c.failed} failed` : null,
            c.running ? `${c.running} running` : null].filter(Boolean).join(" · ") || "—")),
        h("div", { class: "metric" }, h("div", { class: "m-label" }, "Remaining"), h("div", { class: "m-value" }, fmtEta(t.remaining)),
          h("div", { class: "m-sub" }, "critical path p50 (p90)")),
        h("div", { class: "metric" }, h("div", { class: "m-label" }, "Pod cost to finish"),
          h("div", { class: "m-value" }, fmtCostBand(t.pod_cost_remaining_usd)), h("div", { class: "m-sub" }, "p50 (p90)")))),
    h("div", { class: "actionbar" },
      h("button", { class: "btn primary", onclick: async () => (await A()).planDialog(app, take),
        title: "Plan first (no side effects), then approve: the only way a pod is created" }, "Plan & run…"),
      h("button", { class: "btn", onclick: async () => (await A()).planDialog(app, take, { mode: "resume", scope: "all" }),
        title: "Resume the whole pipeline: plan every stage that is not done (stale ones included), to the end, skipping fresh outputs" }, "Resume…"),
      h("button", { class: "btn", onclick: async () => (await A()).planDialog(app, take, { mode: "resume", scope: "all", whatIf: true }),
        title: "What-if (dry-run only): the whole pipeline's order, wall time and pod cost as if your reviews (labels, narration, review gates) were done" }, "What-if…"),
      pipelineRun ? h("button", { class: "btn danger", onclick: async () => (await A()).stopRun(app, pipelineRun) }, "Stop run") : null,
      h("span", { class: "spacer" }),
      h("button", { class: "btn", onclick: async () => (await A()).narrate(app, take),
        title: "Opens the narration review tool (python -m dancecap.moves annotate) in its own window" }, "Narrate moves"),
      h("button", { class: "btn", onclick: async () => (await A()).exportDialog(app, take) }, "Export to Unity"),
      h("button", { class: "btn", onclick: async () => (await A()).openUnity(app) }, "Open Unity"),
      h("button", { class: "btn ghost", onclick: async () => (await A()).loadCaptureDialog(app, take) }, "Load capture"),
      h("button", { class: "btn ghost", onclick: async () => (await A()).consentDialog(app, take) }, "Consent…"),
      h("button", { class: "btn ghost", title: "Open the take's work folder in Explorer", onclick: async () => (await A()).reveal(take, "work") }, "Reveal")),
    h("nav", { class: "tabs", "aria-label": "Take" }, tabs.map(([id, href, label]) => h("a", { href, class: tab === id ? "active" : "" }, label,
      id === "videos" && needCt ? h("span", { class: "count" }, pill(String(needCt), ct.conflict ? "crit" : "warn", "camera types to confirm")) : null,
      id === "runs" && runs.length ? h("span", { class: "count" }, pill(String(runs.length), "info")) : null))));
}

