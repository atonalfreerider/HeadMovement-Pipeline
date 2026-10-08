// Take view (#/take/<take>[/stages|/runs|/analysis|/stage/<id>]): header + tabs; the DAG, the stage list, runs, and
// the analysis lanes / viewer triggers (counterbalance, floorcraft, move labels -> Unity visualisations).
import { h, s, replace, fmtTime, fmtDur, fmtEta, fmtCostBand, fmtClock, fingerprint, prefs, safeRoute } from "../util.js";
import { getx, get, seg, listOf } from "../api.js";
import { statusBadge, statusGlyph, whereBadge, legend, pill } from "../components/status.js";
import { progressBar } from "../components/progress.js";
import { errorText } from "../components/dialog.js";
import { takeHeader } from "./takehead.js";
import { renderDag } from "./dag.js";
import { openStageDrawer } from "./stagedrawer.js";
import { CLASS_SHORT } from "../components/cameraType.js";

export default {
  title: (r) => r.take,
  async mount(root, route, app) {
    const take = route.take;
    let tab = route.tab || "dag";
    let detail = null, snap = null, registry = null;
    let fpHead = null, fpBody = null;
    let drawer = null, drawerId = null;
    let dag = null;
    let showAll = prefs.get("dag_show_all", false);
    let fit = prefs.get("dag_fit", false);
    const headEl = h("div");
    const bodyEl = h("div");
    replace(root, headEl, bodyEl);
    app.registry().then((r) => { registry = r; fpBody = null; drawBody(); }).catch(() => {});

    async function load() {
      try {
        const [d, p] = await Promise.all([getx(`/api/takes/${seg(take)}`), getx(`/api/takes/${seg(take)}/pipeline`)]);
        detail = d.data; snap = p.data;
      } catch (e) {
        replace(bodyEl, h("div", { class: "card crit-text" }, e.status === 404 ? `No take "${take}".` : errorText(e)));
        return;
      }
      const fh = fingerprint({ detail, runs: app.state && app.state.runs_active, tab });
      if (fh !== fpHead) { fpHead = fh; replace(headEl, takeHeader(app, take, detail, tab)); }
      await drawBody();
    }

    // ---------------------------------------------------------------------------------------------- tabs
    let dagHost = null, tabFp = null;
    async function drawBody() {
      if (!snap) return;
      if (tab === "dag") return drawDag();
      if (tab === "stages") return drawStages();
      if (tab === "runs") return drawRuns();
      if (tab === "analysis") return drawAnalysis();
    }

    function gatesBox() {
      const gates = snap.gates_open || [];
      const next = snap.next_actions || [];
      if (!gates.length && !next.length) return null;
      return h("div", { class: "grid2", style: { marginBottom: "12px" } },
        h("div", { class: "card" }, h("h3", { class: "section-title" }, "Needs you"),
          gates.length ? h("div", { class: "gates" }, gates.map((g) => h("div", { class: "gate" },
            h("span", { class: "st-g st-blocked", "aria-hidden": "true" }, "?"), h("span", { style: { flex: 1 } }, h("strong", null, g.stage), " — ", g.message),
            safeRoute(g.link) ? h("a", { class: "btn small", href: safeRoute(g.link) }, "Open") : null)))
            : h("div", { class: "muted" }, "Nothing: no human gate is open.")),
        h("div", { class: "card" }, h("h3", { class: "section-title" }, "Next"),
          next.length ? h("div", { class: "gates" }, next.map((n) => h("div", { class: "row" },
            safeRoute(n.route) ? h("a", { class: "btn small", href: safeRoute(n.route) }, n.label)
              : h("button", { class: "btn small primary", onclick: async () => (await app.actions()).planDialog(app, take, n.plan || {}) }, n.label))))
            : h("div", { class: "muted" }, "—"),
          (snap.branches || []).length ? h("div", { class: "small muted", style: { marginTop: "8px" } }, snap.branches.map((b) =>
            h("div", null, `${b.stage} ${b.label}: `, b.options.map((o) => `${o.name} ${fmtDur(o.eta_p50_s)}`).join(" / ")))) : null));
    }

    function drawDag() {
      const f = fingerprint({ snap, showAll, fit, sel: drawerId, reg: !!registry });
      if (f === fpBody && dagHost && dagHost.isConnected) return;
      fpBody = f;
      const keepScroll = dagHost ? { l: dagHost.scrollLeft, t: dagHost.scrollTop } : null;
      if (!dagHost || !dagHost.isConnected || tabFp !== "dag") {
        tabFp = "dag";
        dagHost = h("div", { class: "dag-wrap", style: { maxHeight: "calc(100vh - 330px)", minHeight: "420px" } });
        const showBox = h("input", { type: "checkbox", checked: showAll });
        showBox.addEventListener("change", () => { showAll = showBox.checked; prefs.set("dag_show_all", showAll); drawDag(); });
        const fitBox = h("input", { type: "checkbox", checked: fit });
        fitBox.addEventListener("change", () => { fit = fitBox.checked; prefs.set("dag_fit", fit); drawDag(); });
        const ctRow = h("div", { class: "dag-ct" });
        replace(bodyEl, gatesBox(),
          h("div", { class: "dag-tools" }, legend(),
            h("span", { class: "spacer" }),
            h("span", { class: "small muted", title: "L this PC · G local 2080 Ti · P Runpod pod · S secure pod · H you · U Unity" },
              h("span", { class: "where" }, "L"), " PC ", h("span", { class: "where where-local_gpu" }, "G"), " 2080 Ti ",
              h("span", { class: "where where-pod" }, "P"), " pod ", h("span", { class: "where where-pod_secure" }, "S"), " secure pod ",
              h("span", { class: "where where-human" }, "H"), " you ", h("span", { class: "where where-unity" }, "U"), " Unity"),
            h("label", { class: "chk small" }, showBox, "show skipped"),
            h("label", { class: "chk small" }, fitBox, "fit width")),
          dagHost,
          h("p", { class: "small muted", style: { marginTop: "6px" } },
            "Columns are phases; rows follow the dependencies. Bold blue = critical path; dashed = soft dependency; violet dashed = iteration loop; tinted bands = one pod session. Hover an edge to see why the order is so. Click a stage for details and actions."),
          ctRow);
      }
      if (dag) dag.destroy();
      dag = renderDag(dagHost, snap, { registry, selected: drawerId, showAll, fit,
        onSelect: (id) => app.navigate(`#/take/${take}/stage/${id}`) });
      if (keepScroll) { dagHost.scrollLeft = keepScroll.l; dagHost.scrollTop = keepScroll.t; }
    }

    function drawStages() {
      const f = fingerprint({ snap, tab });
      if (f === fpBody && tabFp === "stages") return;
      fpBody = f; tabFp = "stages";
      const rows = (snap.stages || []).filter((st) => st.in_variant !== false);
      replace(bodyEl, gatesBox(),
        h("div", { class: "tbl-wrap" }, h("table", { class: "tbl" },
          h("thead", null, h("tr", null, ["#", "stage", "status", "where", "progress / ETA", "estimate", "pod cost", "last run", "camera types", "gates"].map((x) => h("th", null, x)))),
          h("tbody", null, rows.map((st, i) => h("tr", { class: "clickable", onclick: () => app.navigate(`#/take/${take}/stage/${st.id}`) },
            h("td", { class: "muted num" }, String(i + 1)),
            h("td", null, h("strong", null, st.id), h("div", { class: "small muted ellipsis", style: { maxWidth: "320px" } }, st.name || "")),
            h("td", null, statusBadge(st.status, st.status_reason)),
            h("td", null, h("span", { class: "row nowrap" }, whereBadge(st.runs_on), st.session ? h("span", { class: "chip session" }, st.session) : null)),
            h("td", { style: { minWidth: "160px" } }, st.status === "running" ? progressBar(st.progress, { eta: st.eta }) : st.eta ? fmtEta(st.eta) : ""),
            h("td", { class: "num small" }, st.status === "done" ? "" : fmtEta(st.estimate)),
            h("td", { class: "num small" }, st.status === "done" || !(st.cost && st.cost.p50_usd) ? "" : fmtCostBand(st.cost)),
            h("td", { class: "small muted" }, st.last_run ? `${fmtTime(st.last_run.ended)} · rc ${st.last_run.rc ?? "?"}` : ""),
            h("td", { class: "small" }, st.camera_types ? Object.entries(st.camera_types).filter(([, v]) => v && v.length)
              .map(([k, v]) => `${CLASS_SHORT[k] || k} ${v.length}`).join(" · ") : ""),
            h("td", { class: "small" }, (st.gates || []).filter((g) => g.open).map((g) => pill(g.id, "warn", g.message)))))))));
    }

    async function drawRuns() {
      let runs = [];
      try { runs = listOf(await get(`/api/runs?take=${encodeURIComponent(take)}&limit=50`), "runs"); }
      catch (e) { replace(bodyEl, h("p", { class: "crit-text" }, errorText(e))); return; }
      let sessions = [];
      try { sessions = listOf(await get(`/api/sessions?take=${encodeURIComponent(take)}&limit=20`), "sessions"); } catch { /* optional */ }
      const f = fingerprint({ runs, sessions });
      if (f === fpBody && tabFp === "runs") return;
      fpBody = f; tabFp = "runs";
      replace(bodyEl,
        h("h3", { class: "section-title" }, "Runs started by the monitor or the pipeline CLI"),
        runs.length ? runsTable(runs) : h("div", { class: "empty" }, "No runs for this take yet."),
        h("h3", { class: "section-title", style: { marginTop: "16px" } }, "Runpod sessions of this take"),
        sessions.length ? h("div", { class: "tbl-wrap" }, h("table", { class: "tbl compact" },
          h("thead", null, h("tr", null, ["session", "jobs", "state", "origin", "started", "GPU", "cost"].map((x) => h("th", null, x)))),
          h("tbody", null, sessions.map((x) => h("tr", { class: "clickable", onclick: () => app.navigate(`#/runpod/session/${x.sid}`) },
            h("td", { class: "mono small" }, x.sid), h("td", null, (x.jobs || []).join(", ")), h("td", null, pill(x.state, x.state === "ok" ? "good" : x.state === "failed" ? "crit" : x.state === "running" ? "info" : "neutral")),
            h("td", { class: "small muted" }, x.origin), h("td", { class: "small" }, fmtTime(x.started, true)),
            h("td", { class: "small" }, x.gpu || "–"), h("td", { class: "num" }, x.cost_usd !== null && x.cost_usd !== undefined ? "$" + x.cost_usd.toFixed(2) : "–")))))) : h("div", { class: "empty" }, "No sessions."));
    }

    async function drawAnalysis() {
      let a;
      try { a = (await getx(`/api/takes/${seg(take)}/analysis`)).data; }
      catch (e) { replace(bodyEl, h("p", { class: "crit-text" }, errorText(e))); return; }
      const f = fingerprint(a);
      if (f === fpBody && tabFp === "analysis") return;
      fpBody = f; tabFp = "analysis";
      replace(bodyEl, analysisView(app, take, a, detail));
    }

    // ---------------------------------------------------------------------------------------------- drawer
    function syncDrawer(r) {
      const want = r.stage || null;
      if (want === drawerId) return;
      if (drawer) { drawer.destroy(); drawer = null; }
      drawerId = want;
      if (want) drawer = openStageDrawer(app, take, want, () => app.navigate(`#/take/${take}`));
      fpBody = null;
      if (tab === "dag") drawDag();
    }

    await load();
    syncDrawer(route);
    return {
      async refresh() { await load(); if (drawer) drawer.refresh(); },
      route(r) {
        const newTab = r.tab || "dag";
        if (newTab !== tab) { tab = newTab; fpHead = null; fpBody = null; tabFp = null; load(); }
        syncDrawer(r);
      },
      onKey(e) {
        if (e.key === "Escape" && drawer) { app.navigate(`#/take/${take}`); return true; }
        return false;
      },
      destroy() { if (drawer) drawer.destroy(); if (dag) dag.destroy(); if (analysisTip) { analysisTip.remove(); analysisTip = null; } },
    };
  },
};

export function runsTable(runs) {
  return h("div", { class: "tbl-wrap" }, h("table", { class: "tbl compact" },
    h("thead", null, h("tr", null, ["run", "kind", "take", "stages", "state", "progress", "started", "duration", "rc", "origin"].map((x) => h("th", null, x)))),
    h("tbody", null, runs.map((r) => h("tr", { class: "clickable", onclick: () => { location.hash = `#/runs/${r.id}`; } },
      h("td", { class: "mono small" }, r.id), h("td", null, r.kind), h("td", null, r.take || ""),
      h("td", { class: "small ellipsis", style: { maxWidth: "220px" } }, (r.stages || []).join(", ")),
      h("td", null, pill(r.state, { running: "info", done: "good", failed: "crit", lost: "crit", stopped: "warn", waiting_human: "warn", starting: "info", stopping: "warn" }[r.state] || "neutral")),
      h("td", { style: { minWidth: "130px" } }, ["running", "starting"].includes(r.state) ? progressBar(r.progress, { compact: true }) : ""),
      h("td", { class: "small" }, fmtTime(r.started || r.created, true)),
      h("td", { class: "small num" }, r.started ? fmtDur((r.ended || Date.now() / 1000) - r.started) : ""),
      h("td", { class: "small" }, r.rc !== null && r.rc !== undefined ? String(r.rc) : ""),
      h("td", { class: "small muted" }, r.origin || ""))))));
}

// ------------------------------------------------------------------------------------------------ analysis

// Fixed hue order (never cycled); unknown kinds fold into the last neutral slot.
const LANE_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const KIND_ORDER = ["counterbalance", "pivot", "plant", "travel", "rotation", "move", "accent", "physics"];

let analysisTip = null;

function laneColor(kind, i) {
  const k = KIND_ORDER.indexOf(kind);
  return k >= 0 ? LANE_COLORS[k] : i < LANE_COLORS.length ? LANE_COLORS[i] : "#8f8e86";
}

function analysisView(app, take, a, detail) {
  const lanes = a.lanes || [];
  const trig = a.triggers || [];
  const colorOf = new Map();
  lanes.forEach((ln, i) => colorOf.set(ln.kind, laneColor(ln.kind, i)));
  for (const t of trig) if (!colorOf.has(t.kind)) colorOf.set(t.kind, laneColor(t.kind, colorOf.size));
  const all = lanes.flatMap((l) => l.items || []);
  const win = detail && detail.window;
  const t0 = win ? win[0] : all.length ? Math.min(...all.map((x) => x.t0)) : 0;
  const t1 = win ? win[1] : all.length ? Math.max(...all.map((x) => x.t1)) : 1;
  const W = 1100, LBL = 140, LH = 24, PAD = 6;
  const H = 30 + lanes.length * (LH + PAD) + 10;
  const x = (t) => LBL + ((t - t0) / Math.max(1e-6, t1 - t0)) * (W - LBL - 10);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Analysis intervals on the reference clock" });
  const step = (t1 - t0) > 60 ? 10 : 5;
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    svg.appendChild(s("line", { class: "axis-l", x1: x(t), x2: x(t), y1: 18, y2: H - 6 }));
    svg.appendChild(s("text", { class: "axis-t", x: x(t), y: 12, "text-anchor": "middle" }, fmtClock(t).replace(/\.\d$/, "")));
  }
  const tip = h("div", { class: "dag-tip", hidden: true });
  lanes.forEach((ln, i) => {
    const y = 24 + i * (LH + PAD);
    const col = colorOf.get(ln.kind);
    svg.appendChild(s("rect", { class: "lane-bg", x: LBL, y, width: W - LBL - 10, height: LH, rx: 4 }));
    svg.appendChild(s("text", { class: "lane-label", x: 6, y: y + LH / 2 + 4 }, `${ln.kind} (${(ln.items || []).length})`));
    for (const it of ln.items || []) {
      const r = s("rect", {
        class: ["lane-item", it.style === "dashed" ? "dashed" : ""].join(" "), x: x(it.t0), y: y + 3, width: Math.max(2, x(it.t1) - x(it.t0) - 1.5),
        height: LH - 6, rx: 4, fill: col, stroke: col, "fill-opacity": it.confidence === null || it.confidence === undefined ? 0.85 : 0.35 + 0.6 * it.confidence,
      });
      r.addEventListener("mousemove", (e) => {
        tip.hidden = false;
        tip.textContent = `${it.label || ln.kind}  ${fmtClock(it.t0)} – ${fmtClock(it.t1)}${it.confidence !== null && it.confidence !== undefined ? `  confidence ${Math.round(it.confidence * 100)} %` : ""}`;
        tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 400) + "px"; tip.style.top = e.clientY + 14 + "px";
      });
      r.addEventListener("mouseleave", () => { tip.hidden = true; });
      svg.appendChild(r);
    }
  });
  if (analysisTip) analysisTip.remove();
  analysisTip = tip;
  document.body.appendChild(tip);

  const byKind = new Map();
  for (const t of trig) {
    if (!byKind.has(t.kind)) byKind.set(t.kind, []);
    byKind.get(t.kind).push(t);
  }
  const rows = Array.from(byKind.entries()).map(([kind, items]) => {
    const layers = new Set(items.flatMap((t) => t.layers || []));
    const shots = new Set(items.map((t) => t.director && t.director.shot).filter(Boolean));
    const conf = items.map((t) => t.confidence).filter((c) => c !== null && c !== undefined);
    const prov = new Set(items.map((t) => t.provenance).filter(Boolean));
    return { kind, n: items.length, layers: Array.from(layers), shots: Array.from(shots), prov: Array.from(prov),
      conf: conf.length ? conf.reduce((p, c) => p + c, 0) / conf.length : null, src: items[0].source && items[0].source.file };
  });
  const file = a.file || {};
  return h("div", null,
    h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h3", null, "Classifier and analysis intervals"),
        h("span", { class: "muted small" }, `reference clock${win ? `, analysis window ${fmtClock(win[0])} – ${fmtClock(win[1])}` : ""}`)),
      lanes.length ? h("div", { class: "lanes" }, svg) : h("div", { class: "empty" }, "No analysis outputs yet (counterbalance, floorcraft, move labels)."),
      h("p", { class: "small muted" }, "Opacity follows confidence; dashed = below the trigger threshold. Each lane is one trigger kind.")),
    h("div", { class: "card" },
      h("div", { class: "card-head" }, h("h3", null, "Triggers → Unity visualisations"),
        file.exists ? pill(`triggers.json: ${file.n} intervals`, "good") : pill("triggers.json not written yet", "warn"),
        h("button", { class: "btn small", onclick: async () => (await app.actions()).planDialog(app, take, { target: "triggers", scope: "only" }) }, "Run the triggers stage…"),
        h("button", { class: "btn small", onclick: async () => (await app.actions()).exportDialog(app, take) }, "Export to Unity…")),
      h("p", { class: "small muted" }, "When a classifier or analysis marks an interval (e.g. counterbalance), the matching overlay switches on during playback in Unity. Rules: ",
        typeof a.rules === "string" ? a.rules : "taxonomy/triggers.toml", `; threshold ${a.threshold ?? "–"}.`),
      rows.length ? h("table", { class: "tbl compact" },
        h("thead", null, h("tr", null, ["", "trigger kind", "intervals", "switches on (Unity layers)", "director hint", "mean confidence", "provenance", "source"].map((c) => h("th", null, c)))),
        h("tbody", null, rows.map((r, i) => h("tr", null,
          h("td", null, h("span", { style: { display: "inline-block", width: "10px", height: "10px", borderRadius: "2px", background: colorOf.get(r.kind) } })),
          h("td", null, h("strong", null, r.kind)), h("td", { class: "num" }, String(r.n)),
          h("td", { class: "mono small" }, r.layers.join(", ") || "–"), h("td", { class: "small" }, r.shots.join(", ") || "–"),
          h("td", { class: "num" }, r.conf === null ? "–" : `${Math.round(r.conf * 100)} %`), h("td", { class: "small" }, r.prov.join(", ")),
          h("td", { class: "mono small muted" }, r.src || ""))))) : h("div", { class: "empty" }, "No triggers yet."),
      h("div", { class: "small muted", style: { marginTop: "6px" } }, "Sources: ",
        Object.entries(a.sources || {}).map(([k, v]) => `${k} (${(v && v.file) || "?"})`).join(" · ") || "none")));
}
