// Pipeline Monitor shell: hash router, one aggregated /api/state poller (2 s visible, 10 s hidden, ETag), top bar,
// alerts bar, takes sidebar, status bar, keyboard shortcuts. Views live in views/*.js and are loaded on demand.
import { h, replace, fmtTime, fmtEta, prefs, session, fingerprint, safeRoute } from "./util.js";
import * as api from "./api.js";
import { connPill, strip, pill } from "./components/status.js";
import { budgetGauge, budgetFromState, progressBar } from "./components/progress.js";
import { toast, errorText, openDialog, dialogOpen, closeAllDialogs } from "./components/dialog.js";

const VIEWS = {
  farm: "./views/farm.js",
  analysis: "./views/analysis.js",
  snapshots: "./views/snapshots.js",
  dashboard: "./views/dashboard.js",
  wizard: "./views/wizard.js",
  take: "./views/take.js",
  videos: "./views/videos.js",
  runpod: "./views/runpod.js",
  session: "./views/session.js",
  runs: "./views/runs.js",
  logs: "./views/logs.js",
  settings: "./views/settings.js",
  notfound: "./views/notfound.js",
};

export function parseRoute(hash) {
  const parts = (hash || "").replace(/^#\/?/, "").split("/").filter(Boolean).map((p) => {
    try { return decodeURIComponent(p); } catch { return p; }
  });
  if (!parts.length) return { name: "farm" };
  const [a, b, c, d] = parts;
  if (a === "farm") return { name: "farm", dance: b || null };
  if (a === "analysis" && b) return { name: "analysis", run: b };
  if (a === "snapshots") return { name: "snapshots", run: b || null };
  if (a === "legacy") return { name: "dashboard" };
  if (a === "new") return { name: "wizard" };
  if (a === "take" && b) {
    if (!c) return { name: "take", take: b, tab: "dag" };
    if (c === "videos") return { name: "videos", take: b };
    if (c === "stage" && d) return { name: "take", take: b, tab: "dag", stage: d };
    if (["stages", "runs", "analysis"].includes(c)) return { name: "take", take: b, tab: c };
  }
  if (a === "runpod") return b === "session" && c ? { name: "session", sid: c } : { name: "runpod" };
  if (a === "runs") return { name: "runs", id: b || null };
  if (a === "logs") return { name: "logs" };
  if (a === "settings") return { name: "settings" };
  return { name: "notfound" };
}

const routeKey = (r) => `${r.name}|${r.take || ""}|${r.sid || ""}|${r.run || ""}`;
// views that poll their own data and use the full width (no takes sidebar, no /api/state poll)
const SELF_POLLING = new Set(["farm", "analysis", "snapshots"]);

// ------------------------------------------------------------------------------------------------ app object

const app = {
  state: null,
  config: null,
  stateError: null,
  route: null,
  api,
  toast,
  errorText,
  _registry: null,
  _view: null,
  _viewKey: null,
  _subs: new Set(),
  navigate(hash) { if (location.hash !== hash) location.hash = hash; else this.render(); },
  async registry() {
    if (!this._registry) this._registry = api.get("/api/stages").catch((e) => { this._registry = null; throw e; });
    return this._registry;
  },
  onState(fn) { this._subs.add(fn); return () => this._subs.delete(fn); },
  async refreshNow() { await tick(true); },
  takeSummary(take) { return (this.state && this.state.takes || []).find((t) => t.take === take) || null; },
  async actions() { return import("./views/actions.js"); },
  // the farm view polls /api/farm itself: it hands the shell its top-bar and status-bar content and the monitor's identity
  setTop(...nodes) { replace(el.top, nodes); },
  setFarmStatus(items) { farmStatus = items; renderStatus(); },
  noteMonitor(mon) {
    if (!mon) return;
    lastPoll = Date.now();
    app.stateError = null;
    checkInstance({ monitor: mon });
    if (loadedStaticRev === null) loadedStaticRev = mon.static_rev;
    app._mon = mon;
    renderStatus();
  },
};
let farmStatus = null;
window.__dcm = app; // debugging from devtools only

// ------------------------------------------------------------------------------------------------ chrome

const el = {
  top: document.getElementById("top-right"),
  nav: document.getElementById("nav"),
  alerts: document.getElementById("alerts"),
  side: document.getElementById("sidebar"),
  main: document.getElementById("main"),
  status: document.getElementById("statusbar"),
  ribbon: document.getElementById("ribbon"),
};

let loadedStaticRev = null;
let lastPoll = 0;
const dismissed = new Set(session.get("dismissed_alerts", []));

function renderTop() {
  const st = app.state;
  if (!st) { replace(el.top, pill("connecting…", "neutral")); return; }
  const rp = st.runpod || {};
  const pods = rp.pods_running || 0;
  replace(el.top,
    h("a", { href: "#/runpod", class: "top-link", title: "Runpod panel (g r)" },
      connPill(rp.connection),
      pods ? pill(`${pods} pod${pods > 1 ? "s" : ""} running`, "info") : h("span", { class: "muted small" }, "no pods"),
      budgetGauge(budgetFromState(st.budget), { compact: true, title: "Runpod budget" })));
}

function renderNav() {
  const r = app.route || {};
  const active = { farm: "farm", analysis: "farm", snapshots: "farm", dashboard: "legacy", take: "legacy", videos: "legacy", wizard: "legacy", runpod: "legacy",
    session: "legacy", runs: "legacy", logs: "legacy", settings: "settings" }[r.name];
  for (const a of el.nav.querySelectorAll("a")) a.classList.toggle("active", a.dataset.nav === active);
}

function renderAlerts() {
  if (app.route && SELF_POLLING.has(app.route.name)) { el.alerts.hidden = true; return; }   // the farm view shows its own alerts
  const al = (app.state && app.state.alerts || []).filter((a) => !dismissed.has(a.id));
  if (!al.length) { el.alerts.hidden = true; replace(el.alerts); return; }
  el.alerts.hidden = false;
  const order = { red: 0, amber: 1, info: 2 };
  al.sort((a, b) => (order[a.level] ?? 3) - (order[b.level] ?? 3));
  const shown = al.slice(0, 3);
  replace(el.alerts, shown.map((a) => h("div", { class: ["alert", "alert-" + a.level], role: a.level === "red" ? "alert" : null },
    h("span", { class: "alert-icon", "aria-hidden": "true" }, { red: "✕", amber: "!", info: "i" }[a.level] || "i"),
    h("span", { class: "alert-msg" }, a.message),
    safeRoute(a.link) ? h("a", { class: "btn small", href: safeRoute(a.link) }, linkLabel(a)) : null,
    h("button", { class: "icon-btn", title: "Hide for this session", "aria-label": "Dismiss", onclick: () => {
      dismissed.add(a.id); session.set("dismissed_alerts", Array.from(dismissed)); renderAlerts();
    } }, "✕"))),
  al.length > shown.length ? h("div", { class: "alert alert-more muted small" }, `+ ${al.length - shown.length} more`) : null);
}

function linkLabel(a) {
  if (a.source === "camera_types") return "Review videos";
  if (a.source === "budget" || a.source === "pod") return "Runpod";
  return "Open";
}

let sideFp = null;
function renderSidebar(force) {
  const collapsed = prefs.get("side_collapsed", false);
  document.body.classList.toggle("side-collapsed", collapsed);
  const takes = app.state ? app.state.takes || [] : null;
  const cur = app.route && app.route.take;
  const fp = fingerprint({ takes, cur, collapsed });
  if (!force && fp === sideFp) return;
  sideFp = fp;
  const head = h("div", { class: "side-head" },
    collapsed ? null : h("span", { class: "side-title" }, "Takes"),
    collapsed ? null : h("a", { class: "btn small", href: "#/new", title: "New capture (g n)" }, "+ New"),
    h("button", { class: "icon-btn", title: collapsed ? "Show the takes list" : "Hide the takes list",
      "aria-label": "Toggle takes list", onclick: () => { prefs.set("side_collapsed", !collapsed); renderSidebar(true); } },
    collapsed ? "»" : "«"));
  if (collapsed) { replace(el.side, head); return; }
  if (!takes) { replace(el.side, head, h("div", { class: "muted pad" }, "loading…")); return; }
  replace(el.side, head, takes.length ? takes.map((t) => sideTake(t, t.take === cur))
    : h("div", { class: "muted pad" }, "No takes yet. ", h("a", { href: "#/new" }, "Create one")));
}

function sideTake(t, active) {
  if (t.error) return h("a", { class: "side-take err", href: `#/take/${t.take}` }, h("strong", null, t.take),
    h("div", { class: "small crit-text" }, t.error));
  const c = t.counts || {};
  const run = (app.state.runs_active || []).find((r) => r.take === t.take);
  return h("a", { class: ["side-take", active ? "active" : ""], href: `#/take/${t.take}` },
    h("div", { class: "st-row" }, h("strong", { class: "ellipsis" }, t.take),
      h("span", { class: "chip" }, `${t.variant || "?"} · ${t.n_used ?? "?"} cam`)),
    strip(t.strip),
    h("div", { class: "st-row small" },
      h("span", null, `${c.done ?? 0}/${c.total ?? 0} done`),
      t.gates_open ? h("span", { class: "needs" }, `needs you: ${t.gates_open}`) : null),
    run ? progressBar(run.progress, { compact: false, eta: run.progress && run.progress.eta,
      label: `${run.current && run.current.stage || run.kind}` })
      : h("div", { class: "small muted" }, "remaining ", fmtEta(t.remaining)));
}

function reloadButton() {
  return h("button", { class: "btn small primary", title: "Load this window again with the running server's session",
    onclick: () => location.reload() }, "Reload");
}

function renderStatus() {
  const st = app.state;
  const expired = !!staleReason;
  if (app.route && SELF_POLLING.has(app.route.name)) {
    const m = app._mon || {};
    const updated = m.static_rev && loadedStaticRev && m.static_rev !== loadedStaticRev;
    replace(el.status, ...(farmStatus || []), h("span", { class: "sb-spacer" }),
      expired ? h("span", { class: "sb-item crit-text" }, `${staleReason}: reload this window`) : null,
      expired ? reloadButton() : updated ? h("button", { class: "btn small primary", onclick: () => location.reload() }, "UI updated — reload") : null,
      h("span", { class: "sb-item muted" }, `updated ${fmtTime(lastPoll / 1000)} · monitor :${location.port} v${m.version || "?"}`));
    el.ribbon.hidden = true;
    return;
  }
  if (!st) {
    replace(el.status, h("span", { class: app.stateError ? "crit-text" : "muted" },
      expired ? `This window must reload: ${staleReason}. `
        : app.stateError ? `Backend not answering (${app.stateError}) — retrying…` : "Connecting to the monitor backend…"),
    expired ? reloadButton() : null);
    return;
  }
  const locks = st.locks || {};
  const runs = st.runs_active || [];
  const m = st.monitor || {};
  if (loadedStaticRev === null) loadedStaticRev = m.static_rev;
  const updated = m.static_rev && loadedStaticRev && m.static_rev !== loadedStaticRev;
  replace(el.status,
    h("a", { href: "#/runs", class: "sb-item" }, `${runs.length} run${runs.length === 1 ? "" : "s"} active`),
    h("span", { class: "sb-item", title: locks.gpu && locks.gpu.owner || "" },
      "GPU lock: ", locks.gpu && locks.gpu.held ? h("span", { class: "warn-text" }, "held") : "free"),
    h("span", { class: "sb-item ellipsis", title: locks.unity && locks.unity.owner || "" },
      "Unity lock: ", locks.unity && locks.unity.held ? h("span", { class: "warn-text" }, "held (Unity work in progress)") : "free"),
    h("span", { class: "sb-item", title: "L0 = parsed from logs; L1 = live GPU/upload probes; L2 = instrumented stages" },
      `progress level ${st.runpod && st.runpod.progress_level || "L0"}`),
    h("span", { class: "sb-item", title: "heavy CV / neural work runs on Runpod; this PC does orchestration, the UI and light checks" },
      "compute: heavy → Runpod"),
    h("span", { class: "sb-spacer" }),
    expired ? h("span", { class: "sb-item crit-text" }, `${staleReason}: reload this window`)
      : app.stateError ? h("span", { class: "sb-item crit-text" }, `connection lost (${app.stateError}) — retrying`) : null,
    expired ? reloadButton()
      : updated ? h("button", { class: "btn small primary", onclick: () => location.reload() }, "UI updated — reload") : null,
    h("span", { class: "sb-item muted" }, `updated ${fmtTime(lastPoll / 1000)} · monitor :${location.port} v${m.version || "?"}`));
  el.ribbon.hidden = !(m.mock || m.demo);
  if (m.mock || m.demo) el.ribbon.textContent = m.demo_label || "DEMO DATA (mock server) — not your live pipeline";
}

// ------------------------------------------------------------------------------------------------ stale window
// After the backend restarts (Settings > Restart, the idle exit, a crash and a relaunch) an open window still holds
// the earlier server's token and cookie: every GET answers 403, or, once a new window has set the new cookie, the
// polls work and every write fails. The window notices (a 403 from the auth guard, or /api/state reporting another
// server instance) and reloads itself ONCE to pick up the new session: never in a loop (at most once per 10 min),
// never under an open dialog and never while you type (then it asks, and tries again on the next poll).
const RELOAD_KEY = "dcm_auto_reload_at";
let bootInstance = null;
let staleReason = null;
let staleToast = null;

function sessionStale(reason) {
  if (!staleReason) { staleReason = reason; renderStatus(); }
  let last = 0;
  try { last = Number(sessionStorage.getItem(RELOAD_KEY) || 0); } catch { /* storage blocked */ }
  const a = document.activeElement;
  const typing = !!a && (/^(INPUT|TEXTAREA|SELECT)$/.test(a.tagName || "") || !!a.isContentEditable);
  if (Date.now() - last > 10 * 60 * 1000 && !dialogOpen() && !typing) {
    try { sessionStorage.setItem(RELOAD_KEY, String(Date.now())); } catch { /* storage blocked */ }
    console.info(`[monitor] reloading this window: ${reason}`);
    location.reload();
    return;
  }
  if (!staleToast || !staleToast.isConnected) {
    staleToast = toast(`This window must reload to keep working: ${staleReason}. ` +
      "Nothing saved is lost; the new-capture wizard keeps its draft.", "warn",
    { timeout: 0, action: { label: "Reload now", onClick: () => location.reload() } });
  }
}
window.addEventListener("dcm:auth-expired", () => sessionStale("the server no longer accepts this window's session"));

function checkInstance(st) {
  const inst = st && st.monitor && st.monitor.instance;
  if (!inst) return;
  if (!bootInstance) bootInstance = inst;
  else if (inst !== bootInstance) sessionStale("the monitor server restarted");
}

// ------------------------------------------------------------------------------------------------ polling

let inflight = false;
let timer = null;
let lastViewError = null;
async function tick(force) {
  if (app.route && SELF_POLLING.has(app.route.name)) { renderStatus(); return; }   // the farm view polls /api/farm itself
  if (inflight && !force) return;
  inflight = true;
  try {
    let res;
    try {
      res = await api.getx("/api/state");
    } catch (e) {
      app.stateError = e instanceof TypeError ? "offline" : (e.code || e.message);
      if (api.isAuthError(e)) app.stateError = "session expired";
      renderStatus();
      return;
    }
    const { data, changed } = res;
    lastPoll = Date.now();
    const wasErr = app.stateError;
    app.stateError = null;
    if (changed || wasErr || !app.state) {
      app.state = data;
      renderTop(); renderAlerts(); renderSidebar();
      for (const fn of app._subs) { try { fn(data); } catch (e) { console.error(e); } }
    }
    checkInstance(app.state);
    renderStatus();
    if (viewLoadFailed && !dialogOpen()) {
      let last = 0;
      try { last = Number(sessionStorage.getItem("dcm_view_reload_at") || 0); } catch { /* storage blocked */ }
      if (Date.now() - last > 60 * 1000) {
        try { sessionStorage.setItem("dcm_view_reload_at", String(Date.now())); } catch { /* storage blocked */ }
        console.info("[monitor] reloading this window: a view failed to load while the server was down");
        location.reload();
        return;
      }
    }
    // a view's own rendering bug is reported as such, never as "backend not answering"
    if (app._view && app._view.refresh) {
      try {
        await app._view.refresh(app.state, { changed });
        lastViewError = null;
      } catch (e) {
        console.error(e);
        const msg = errorText(e);
        if (msg !== lastViewError) { lastViewError = msg; toast(`This view could not refresh: ${msg}`, "warn"); }
      }
    }
  } finally {
    inflight = false;
  }
}

function schedule() {
  clearTimeout(timer);
  const visible = document.visibilityState === "visible";
  const base = (app.config && app.config.settings && app.config.settings.poll_ms) || 2000;
  const ms = app.stateError ? 2000 : visible ? Math.max(1000, base) : 10000;
  timer = setTimeout(async () => { await tick(); schedule(); }, ms);
}
document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") { tick(); schedule(); } });

// ------------------------------------------------------------------------------------------------ routing

let renderSeq = 0;
app.render = async function render() {
  const route = parseRoute(location.hash);
  const key = routeKey(route);
  app.route = route;
  document.body.classList.toggle("farm-mode", SELF_POLLING.has(route.name));
  renderNav();
  renderSidebar();
  if (!SELF_POLLING.has(route.name) && !app.state && !inflight) tick(true);   // older tools need the aggregated /api/state
  if (app._view && app._viewKey === key && app._view.route) {
    app._view.route(route);
    return;
  }
  const seq = ++renderSeq;
  if (app._view && app._view.destroy) { try { app._view.destroy(); } catch (e) { console.error(e); } }
  app._view = null;
  app._viewKey = key;
  replace(el.main, h("div", { class: "loading" }, "loading…"));
  try {
    const mod = await import(VIEWS[route.name] || VIEWS.notfound);
    if (seq !== renderSeq) return;
    const root = h("div", { class: "view view-" + route.name });
    replace(el.main, root);
    const v = await mod.default.mount(root, route, app);
    if (seq !== renderSeq) { if (v && v.destroy) v.destroy(); return; }
    app._view = v || {};
    document.title = (mod.default.title ? mod.default.title(route) + " · " : "") + "Pipeline Monitor";
    el.main.focus({ preventScroll: true });
  } catch (e) {
    console.error(e);
    // a module fetch that failed because the server was down (restarting) stays failed for this page's lifetime
    // (browsers remember failed imports): the next successful poll reloads the page once (see tick)
    const network = e instanceof TypeError;
    if (network) viewLoadFailed = true;
    replace(el.main, h("div", { class: "pad" }, h("h2", null, "This view failed to load"),
      h("p", { class: "crit-text" }, network
        ? "The monitor server did not answer while this page loaded (it may have been restarting). The page reloads itself when the server answers again."
        : errorText(e)),
      h("button", { class: "btn primary", onclick: () => location.reload() }, "Reload")));
  }
};
let viewLoadFailed = false;

window.addEventListener("hashchange", () => { closeAllDialogs(); app.render(); });

// ------------------------------------------------------------------------------------------------ keyboard

const CHORDS = { f: "#/", t: "#/legacy", n: "#/new", l: "#/logs", s: "#/settings" };
const MODIFIER_KEYS = new Set(["Shift", "CapsLock", "Control", "Alt", "AltGraph", "Meta", "OS"]);
// focusable controls whose Enter / Space keep their native meaning (press THAT control)
const CONTROL_SEL = "button, a[href], video, audio, summary, [contenteditable], [role=button], [role=radio], " +
  "[role=checkbox], [role=switch], [role=tab], [role=menuitem], [role=option], [role=slider], [role=link]";
const isTextEntry = (t) => !!t && (/^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName || "") || !!t.isContentEditable);
const isControl = (t) => !!(t && t.closest && t.closest(CONTROL_SEL));

let gPending = 0;
document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (MODIFIER_KEYS.has(e.key)) return; // Shift alone keeps a pending chord alive
  if (isTextEntry(e.target)) return;
  if (dialogOpen()) { gPending = 0; return; }
  const now = Date.now();
  // 1. A pending "g" chord comes FIRST: its second key never reaches the view ("g t" = go to takes, never Video
  //    Review's T = tripod; "g n" = new capture, never N = narrate). Any other key just ends the chord.
  if (gPending && now - gPending < 1200) {
    gPending = 0;
    e.preventDefault();
    if (CHORDS[e.key]) app.navigate(CHORDS[e.key]);
    return;
  }
  gPending = 0;
  if (e.key === "g") { if (!e.repeat) gPending = now; e.preventDefault(); return; }
  if (e.key === "?") { e.preventDefault(); showHelp(); return; }
  // 2. Enter / Space on a focused button, link, player, radio... press that control, never a view shortcut.
  if ((e.key === "Enter" || e.key === " " || e.key === "Spacebar") && isControl(e.target)) return;
  if (app._view && app._view.onKey && app._view.onKey(e)) e.preventDefault();
});

function showHelp() {
  const rows = [
    ["g f", "farm (the defrag map)"], ["g t", "older tools: takes"], ["g n", "new capture"], ["g l", "logs"],
    ["g s", "settings"], ["Arrows", "farm map: move between seconds"], ["?", "this help"], ["Esc", "close a drawer or the top dialog"],
    ["J / K", "Video Review: next / previous video"], ["H / T", "Video Review: handheld / tripod (saves)"],
    ["Z", "Video Review: toggle zoom (saves)"], ["A / Enter", "Video Review: accept / confirm the selected video"],
    ["Space", "Video Review: play / pause"], ["N", "Video Review: narrate with this camera"],
    ["Enter / Space", "on a focused button: press that button"],
  ];
  openDialog({ title: "Keyboard shortcuts", body: h("table", { class: "tbl compact" },
    h("tbody", null, rows.map(([k, v]) => h("tr", null, h("td", null, h("kbd", null, k)), h("td", null, v))))) });
}

// ------------------------------------------------------------------------------------------------ one window
// (2026-10-22) There is one monitor window: a launch or a new job writes work/monitor/focus.json (monitor.set_focus) and
// this window follows it instead of another window opening. A fresh window only notes the current focus: its own URL
// already shows what it was opened for.
let focusTs = null;
async function followFocus() {
  let f;
  try { f = await api.get("/api/focus"); } catch { return; }   // the state poll reports a server that is down
  const ts = (f && f.ts) || 0;
  if (focusTs === null) { focusTs = ts; return; }
  if (ts <= focusTs) return;
  focusTs = ts;
  if (f.route && f.route.startsWith("#/") && location.hash !== f.route) {
    location.hash = f.route;
    toast(`Now showing ${f.reason || f.route}`, "info");
  }
}

// ------------------------------------------------------------------------------------------------ boot

(async function boot() {
  const theme = prefs.get("theme", null);
  if (theme) document.documentElement.dataset.theme = theme;
  if (!api.TOKEN) {
    toast("This page has no monitor token: open it from the Pipeline Monitor launcher (or reload).", "warn", { timeout: 0 });
  }
  try {
    app.config = await api.get("/api/config");
    const t = app.config.settings && app.config.settings.theme;
    if (t && t !== "system") { document.documentElement.dataset.theme = t; prefs.set("theme", t); }
    else if (t === "system") { delete document.documentElement.dataset.theme; prefs.set("theme", null); }
  } catch (e) { /* the poller reports connection problems */ }
  app.route = parseRoute(location.hash);   // so the first tick knows whether the farm view (no /api/state) is the home
  renderTop(); renderStatus(); renderSidebar(true);
  const first = tick(true); // the top bar and the takes list do not wait for the view (cold server: several seconds)
  await app.render();
  await first;
  schedule();
  followFocus();
  setInterval(followFocus, 3000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) followFocus(); });
})();

export default app;
