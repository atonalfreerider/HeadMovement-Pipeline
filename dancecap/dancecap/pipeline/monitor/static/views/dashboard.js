// Dashboard (#/): every take with its status strip, progress, remaining ETA and pod cost; the Runpod card, active
// runs and locks on the right.
import { h, replace, fmtEta, fmtCostBand, fmtAgo, fingerprint } from "../util.js";
import { strip, connPill, pill } from "../components/status.js";
import { progressBar, budgetGauge, budgetFromState } from "../components/progress.js";
import { getx } from "../api.js";
import { activeSnapshots, sessionLive } from "./runpod.js";

const VARIANT = { multi: "multi-view", two: "two-view", single: "single-view" };

export function camTypeSummary(ct) {
  if (!ct) return null;
  const parts = [];
  if (ct.confirmed) parts.push(pill(`${ct.confirmed} confirmed`, "good"));
  if (ct.suggested) parts.push(pill(`${ct.suggested} suggested`, "warn"));
  if (ct.unset) parts.push(pill(`${ct.unset} unset`, "neutral", "no camera type yet: the general model (handheld + zoom) is used"));
  if (ct.conflict) parts.push(pill(`${ct.conflict} conflict`, "crit"));
  return h("span", { class: "row", style: { gap: "4px" } }, parts);
}

function takeCard(app, t) {
  if (t.error) {
    return h("div", { class: "take-card" }, h("div", { class: "tc-head" }, h("span", { class: "tc-title" }, t.take)),
      h("p", { class: "crit-text" }, "Could not read this take: ", t.error));
  }
  const c = t.counts || {};
  const runs = (app.state.runs_active || []).filter((r) => r.take === t.take);
  const issues = [];
  if (c.failed) issues.push(h("span", { class: "crit-text" }, `${c.failed} failed`));
  if (c.stale) issues.push(h("span", { class: "warn-text" }, `${c.stale} stale`));
  if (c.running) issues.push(h("span", { class: "st-running" }, `${c.running} running`));
  const go = () => app.navigate(`#/take/${t.take}`);
  return h("div", { class: "take-card", role: "link", tabindex: "0", style: { cursor: "pointer" }, onclick: go,
    onkeydown: (e) => { if (e.key === "Enter") go(); } },
    h("div", { class: "tc-head" },
      h("span", { class: "tc-title" }, t.title || t.take),
      h("span", { class: "chip mono" }, t.take),
      h("span", { class: "muted" }, [t.level, t.category].filter(Boolean).join(" / ")),
      h("span", { class: "spacer" }),
      h("span", { class: "chip" }, `${VARIANT[t.variant] || t.variant} · ${t.n_used ?? "?"} of ${t.n_videos ?? "?"} videos`)),
    h("div", { class: "tc-strip" }, strip(t.strip, { onClick: (id) => app.navigate(`#/take/${t.take}/stage/${id}`) })),
    h("div", { class: "tc-grid" },
      h("div", null, h("span", { class: "flabel" }, "Stages"),
        h("div", null, h("strong", null, `${c.done ?? 0} / ${c.total ?? 0}`), " done"),
        h("div", { class: "row small", style: { gap: "8px" } }, issues.length ? issues : h("span", { class: "muted" }, "no problems"))),
      h("div", null, h("span", { class: "flabel" }, "Remaining"), h("div", null, fmtEta(t.remaining)),
        h("div", { class: "small muted" }, "critical path, p50 (p90)")),
      h("div", null, h("span", { class: "flabel" }, "Pod cost to finish"), h("div", null, fmtCostBand(t.pod_cost_remaining_usd)),
        h("div", { class: "small muted" }, "p50 (p90)")),
      h("div", null, h("span", { class: "flabel" }, "Camera types"), camTypeSummary(t.camera_types),
        t.gates_open ? h("div", { class: "small needs" }, `needs you: ${t.gates_open}`) : h("div", { class: "small muted" }, `last activity ${fmtAgo(t.last_activity)}`))),
    runs.map((r) => h("div", { style: { marginTop: "10px" } },
      progressBar(r.progress, { eta: r.progress && r.progress.eta,
        label: [h("strong", null, r.kind), ` ${r.current && r.current.stage ? r.current.stage : (r.stages || []).join(", ")}`,
          r.current && r.current.step ? h("span", { class: "muted" }, ` · ${r.current.step}`) : null] }))),
    h("div", { class: "tc-actions", onclick: (e) => e.stopPropagation() },
      h("a", { class: "btn small", href: `#/take/${t.take}` }, "Pipeline"),
      h("a", { class: "btn small", href: `#/take/${t.take}/videos` }, "Videos & camera types"),
      h("button", { class: "btn small", onclick: async () => (await app.actions()).planDialog(app, t.take) }, "Plan & run…"),
      h("button", { class: "btn small", onclick: async () => (await app.actions()).narrate(app, t.take) }, "Narrate moves"),
      h("a", { class: "btn small ghost", href: `#/take/${t.take}/analysis` }, "Triggers")));
}

function runpodCard(app, live) {
  const st = app.state;
  const rp = st.runpod || {};
  const pods = (live && live.rp && live.rp.pods) || [];
  return h("div", { class: "card" },
    h("div", { class: "card-head" }, h("h3", null, "Runpod"), connPill(rp.connection)),
    h("div", { class: "kv", style: { marginBottom: "10px" } },
      h("span", null, "pods running"), h("span", null, String(rp.pods_running || 0)),
      h("span", null, "sessions active"), h("span", null, String(rp.sessions_active || 0)),
      h("span", null, "progress level"), h("span", null, rp.progress_level || "L0"),
      rp.message ? h("span", null, "message") : null, rp.message ? h("span", { class: "warn-text" }, rp.message) : null),
    pods.map((p) => {
      const sn = p.session && live.snaps[p.session.sid];
      return h("a", { class: "dash-session", href: p.session ? `#/runpod/session/${p.session.sid}` : "#/runpod" },
        h("div", { class: "row small" }, h("strong", null, p.name || p.id), h("span", { class: "muted" }, `$${(p.cost_per_hr || 0).toFixed(2)}/h`),
          p.session && p.session.take ? h("span", null, p.session.take) : null, h("span", { class: "spacer" }), connPill(p.connection || p.status, true)),
        sn ? sessionLive(sn, { compact: true }) : null);
    }),
    h("div", { style: { marginTop: "10px" } }, budgetGauge(budgetFromState(st.budget), { title: "Budget" })),
    h("p", { class: "small muted", style: { marginTop: "10px" } },
      "Read-only: the monitor never creates, stops or deletes a pod on its own. Pods start only from an approved plan."),
    h("a", { class: "btn small", href: "#/runpod" }, "Open the Runpod panel"));
}

function runsCard(app) {
  const runs = app.state.runs_active || [];
  return h("div", { class: "card" },
    h("div", { class: "card-head" }, h("h3", null, "Active runs"), h("a", { class: "small", href: "#/runs" }, "all runs")),
    runs.length ? runs.map((r) => h("a", { href: `#/runs/${r.id}`, style: { display: "block", color: "inherit", marginBottom: "8px" } },
      h("div", { class: "row small" }, h("strong", null, r.kind), h("span", null, r.take || ""), h("span", { class: "spacer" }),
        pill(r.state, r.state === "running" ? "info" : r.state === "waiting_human" ? "warn" : "neutral")),
      progressBar(r.progress, { eta: r.progress && r.progress.eta, compact: false,
        label: r.current && r.current.stage ? r.current.stage : (r.stages || []).join(", ") })))
      : h("div", { class: "muted" }, "Nothing running."));
}

function locksCard(app) {
  const l = app.state.locks || {};
  return h("div", { class: "card" }, h("div", { class: "card-head" }, h("h3", null, "This PC")),
    h("div", { class: "kv" },
      h("span", null, "GPU (2080 Ti)"), h("span", null, l.gpu && l.gpu.held ? h("span", { class: "warn-text", title: l.gpu.owner || "" }, "in use: " + (l.gpu.owner || "")) : "free"),
      h("span", null, "Unity"), h("span", { class: "ellipsis", title: l.unity && l.unity.owner || "" }, l.unity && l.unity.held ? h("span", { class: "warn-text" }, "lock held: " + (l.unity.owner || "")) : "free"),
      h("span", null, "policy"), h("span", { class: "small" }, "2 local analyses at a time · 1 local GPU stage · 1 pod session · heavy CV runs on Runpod")));
}

export default {
  title: () => "Takes",
  async mount(root, route, app) {
    let fp = null;
    let live = { rp: null, snaps: {} };
    async function loadLive() {
      const r = app.state && app.state.runpod;
      if (!r || !(r.pods_running || r.sessions_active)) { live = { rp: null, snaps: {} }; return; }
      try {
        const rp = (await getx("/api/runpod")).data;
        live = { rp, snaps: await activeSnapshots(rp) };
      } catch { /* keep the last view */ }
    }
    function render(force) {
      const st = app.state;
      if (!st) { replace(root, h("div", { class: "loading" }, "Waiting for the backend…")); return; }
      const f = fingerprint({ t: st.takes, r: st.runs_active, rp: st.runpod, b: st.budget, l: st.locks, live });
      if (!force && f === fp) return;
      fp = f;
      const takes = st.takes || [];
      replace(root,
        h("div", { class: "page-head" }, h("h1", null, "Takes"),
          h("span", { class: "muted" }, `${takes.length} take${takes.length === 1 ? "" : "s"}`),
          h("a", { class: "btn primary", href: "#/new" }, "+ New capture")),
        h("div", { class: "grid-main-side" },
          h("div", null, takes.length ? takes.map((t) => takeCard(app, t))
            : h("div", { class: "empty" }, "No takes yet. ", h("a", { href: "#/new" }, "Start a new capture"),
              " (videos + soundtrack + dancers).")),
          h("div", null, runpodCard(app, live), runsCard(app), locksCard(app))));
    }
    render(true);
    loadLive().then(() => render(false));
    return { async refresh() { await loadLive(); render(false); } };
  },
};
