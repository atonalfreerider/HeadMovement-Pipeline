// The farm view (#/ and #/farm/<dance>): the DEFRAG MAP of the dance's final result and, below it, the compact panels (pods, active
// chunk jobs, queue, budget, volume, world). Reads one bundle, GET /api/farm, every 2 s (ETag: nothing new = 304). docs/FARM.md 7.
import { h, replace, fmtDur, fmtUSD, fmtAgo, fmtTime, fingerprint, prefs, clamp } from "../util.js";
import { Defrag, chunkColor } from "../components/defrag.js";
import { openDialog, confirmDialog, typedConfirm, toast, errorText } from "../components/dialog.js";
import { pill } from "../components/status.js";
import * as api from "../api.js";

const POLL_VISIBLE_MS = 2000, POLL_HIDDEN_MS = 10000;
const STAGE_TEXT = { none: "nothing yet", front: "front only (cameras, 2D, sync; no 3D yet)", fit: "raw 3D fit", refined: "refined, awaiting stitch", analysed: "refined and analysed" };
const STATE_TEXT = { complete: "complete", partial: "partial", none: "not done" };
const GPU_SHORT = (g) => (g || "?").replace(/^NVIDIA /, "").replace(/GeForce /, "").replace(/ ?80GB( HBM3)?/, "").replace(/-SXM4/, " SXM").replace(/ Generation/, "");

const fmtSpan = (a, b) => (a == null || b == null ? "–" : `${a.toFixed(1)} → ${b.toFixed(1)} s`);
const fmtPct = (p) => (p > 0 && p < 1 ? "<1" : String(Math.round(p || 0)));
const fmtMin = (m) => (m == null ? "–" : fmtDur(m * 60));
const mmss = (s) => { const m = Math.floor(Math.max(0, s) / 60), r = Math.floor(Math.max(0, s) % 60); return `${m}:${String(r).padStart(2, "0")}`; };

function etaText(eta) {
  if (!eta || eta.p50 == null) return null;
  const rng = eta.p10 != null && eta.p90 != null ? ` (range ${fmtDur(eta.p10, { short: true })} to ${fmtDur(eta.p90, { short: true })})` : "";
  return { text: `ETA ${fmtDur(eta.p50)}`, title: `remaining time, p50${rng}` };
}

function glyphFor(st) {
  return { done: "✓", running: "▶", pending: "○", failed: "✕", skipped: "–", queued: "◷" }[st] || "·";
}

function stepChips(steps) {
  return h("span", { class: "fm-steps", "aria-label": "steps" }, (steps || []).map((s) =>
    h("span", { class: ["fm-step", "fm-step-" + s.status], title: [s.name, s.status, s.step, s.detail].filter(Boolean).join(" · ") },
      h("span", { class: "fm-step-g", "aria-hidden": "true" }, glyphFor(s.status)), s.name)));
}

function bar(frac, { running = true, title = null } = {}) {
  const known = frac != null && isFinite(frac);
  return h("div", { class: ["fm-bar", running ? "running" : "", known ? "" : "indet"], title,
    role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": known ? String(Math.round(frac * 100)) : null },
  h("div", { class: "fm-bar-fill", style: { width: known ? `${clamp(frac, 0, 1) * 100}%` : "35%" } }));
}

function srcPill(src) {
  return src === "fit_full" ? pill("fit_full", "info", "a dancecap.fit_full session (started outside the farm); the monitor only reads it")
    : src === "other" ? pill("not ours", "warn", "billing on the account, not started by the farm or by fit_full")
      : pill("farm", "good", "started by the farm launcher");
}

// ------------------------------------------------------------------------------------------------ the view

export default {
  title: () => "Farm",
  async mount(root, route, app) {
    const S = { dance: route.dance || prefs.get("farm_dance", null), data: null, etagBusy: false, timer: null, destroyed: false,
      lastFp: {}, selected: -1, error: null, lastOk: 0, seq: 0, fetchedAt: Date.now() };
    // the server's clock now, estimated from the last bundle: ages are computed from absolute timestamps so they keep ticking
    // between bundles (an unchanged bundle answers 304 and is not rebuilt)
    const srvNow = () => (S.data ? S.data.now + (Date.now() - S.fetchedAt) / 1000 : Date.now() / 1000);

    // ---- skeleton
    const danceSel = h("select", { "aria-label": "Dance", title: "The dance whose final result is shown", onchange: () => go(danceSel.value) });
    const updatedEl = h("span", { class: "muted small fm-updated" });
    const worldEl = h("span", { class: "fm-world" });
    const watcherEl = h("span", { class: "fm-world" });
    const btn = (label, cls, onclick, title) => h("button", { class: ["btn", cls || ""], onclick, title }, label);
    const bFill = btn("Fill gaps", "primary", () => fillGaps(), "Plan the missing parts as 30-second chunks, show cost and time, and start pods only after you confirm");
    const bPull = btn("Pull now", "", () => act("/api/farm/pull", "Pull now"), "Download every finished chunk from the volume now");
    const bLog = btn("Open watcher log", "", () => openLog(), "The watcher's log (farm/watcher.log)");
    const bRestart = btn("Restart watcher", "", () => restartWatcher(), "Stop the detached watcher and start it again");
    const bStop = btn("Stop pods", "danger", () => stopPods(), "Delete this farm's pods (asks first)");
    const bPng = btn("Export map PNG", "", () => exportPng(), "Save the map as it is drawn now to farm/review/");
    const head = h("div", { class: "fm-head" },
      h("div", { class: "fm-head-l" }, h("h1", null, "Final result"), danceSel, worldEl),
      h("div", { class: "fm-actions" }, bFill, bPull, bLog, bRestart, bStop, bPng));
    const subhead = h("div", { class: "fm-subhead" }, watcherEl, updatedEl);
    const alertsEl = h("div", { class: "fm-alerts" });
    const heroEl = h("div", { class: "fm-hero" });
    const canvas = h("canvas", { class: "fm-canvas", tabindex: "0", role: "img",
      "aria-label": "Defrag map of the final result: one square per second, green complete, hatched amber partial, dark not done. Arrow keys move the selection." });
    const mapWrap = h("div", { class: "fm-canvas-wrap" }, canvas);
    const tip = h("div", { class: "fm-tip", role: "tooltip", hidden: true });
    const detail = h("div", { class: "fm-detail", "aria-live": "polite" }, h("span", { class: "muted" }, "Hover or click a second for its stage, cameras and reasons."));
    const legendEl = h("div", { class: "fm-legend" });
    const chunkLegend = h("div", { class: "fm-chunks" });
    const mapCard = h("section", { class: "card fm-mapcard" },
      h("div", { class: "fm-maphead" }, h("h2", null, "Timeline, one square per second"), h("span", { class: "spacer" }), legendEl),
      mapWrap, detail, chunkLegend);
    const panels = {
      pods: h("section", { class: "card fm-panel" }), jobs: h("section", { class: "card fm-panel fm-wide" }),
      queue: h("section", { class: "card fm-panel" }), budget: h("section", { class: "card fm-panel" }),
      volume: h("section", { class: "card fm-panel" }), world: h("section", { class: "card fm-panel" }),
      events: h("section", { class: "card fm-panel fm-wide" }),
    };
    replace(root, head, subhead, alertsEl, heroEl, mapCard, tip,
      h("div", { class: "fm-grid" }, panels.jobs, panels.pods, panels.queue, panels.budget, panels.volume, panels.world, panels.events),
      h("p", { class: "muted small fm-foot" }, "Older tools (takes, camera types, new capture): ", h("a", { href: "#/legacy" }, "open"),
        " · the old job lists were archived to work/pipeline/archive/ and are not shown here."));

    // ---- the map
    const map = new Defrag(canvas, {
      onHover(i, x, y) {
        if (i < 0 || !S.data) { tip.hidden = true; return; }
        showTip(i, x, y);
      },
      onSelect(i) { S.selected = i; showDetail(i); tip.hidden = true; },
    });

    function cellText(c, d) {
      const ch = d.chunks.find((x) => x.id === c.c);
      const lines = [`${mmss(c.i * (d.coverage.cell_s || 1))} to ${mmss((c.i + 1) * (d.coverage.cell_s || 1))} of the timeline` +
        (c.t0 != null ? `  (reference clock ${c.t0.toFixed(1)} to ${c.t1.toFixed(1)} s)` : "")];
      lines.push(`${STATE_TEXT[c.s]} · ${STAGE_TEXT[c.g] || c.g}`);
      if (c.n != null) lines.push(`cameras ${c.n}${c.m != null ? " of " + c.m : ""}${c.f ? " · " + c.f : ""}`);
      if (c.r && c.r.length) lines.push(c.r.join("; "));
      if (c.c) lines.push(`chunk ${c.c}${ch ? ` (${ch.status}${ch.source === "fit_full" ? ", fit_full" : ""})` : ""}${c.live ? " · being worked on now" : ""}`);
      if (c.o && c.o.length) lines.push(`overlap of ${c.o.join(", ")}`);
      return lines;
    }

    function showTip(i, x, y) {
      const c = S.data.coverage.cells[i];
      if (!c) { tip.hidden = true; return; }
      const lines = cellText(c, S.data);
      replace(tip, lines.map((l, k) => h("div", { class: k === 0 ? "fm-tip-h" : "" }, l)));
      tip.hidden = false;
      const w = tip.offsetWidth, hh = tip.offsetHeight;
      const px = Math.min(window.innerWidth - w - 8, x + 14), py = y + 18 + hh > window.innerHeight ? y - hh - 14 : y + 18;
      tip.style.left = `${Math.max(4, px)}px`;
      tip.style.top = `${Math.max(4, py)}px`;
    }

    function showDetail(i) {
      const d = S.data;
      if (!d || i < 0 || !d.coverage.cells[i]) {
        replace(detail, h("span", { class: "muted" }, "Hover or click a second for its stage, cameras and reasons."));
        return;
      }
      const c = d.coverage.cells[i];
      replace(detail, h("span", { class: ["fm-dot", "fm-dot-" + c.s] }), cellText(c, d).map((l, k) => h("span", { class: k === 0 ? "fm-detail-h" : "" }, l)));
    }

    // ---- rendering of one bundle
    function renderAll(d) {
      S.data = d;
      S.dance = d.dance;
      prefs.set("farm_dance", d.dance);   // a dance that is only open through its route (a demo) is remembered too
      const cov = d.coverage;
      const chunkIndex = {};
      d.chunks.forEach((c, k) => { chunkIndex[c.id] = k; });
      map.setModel({ cells: cov.cells, chunkIndex, cellS: cov.cell_s || 1 });
      if (S.selected >= 0) showDetail(S.selected);
      renderHead(d); renderAlerts(d); renderHero(d); renderLegends(d);
      const fp = (k, o) => { const f = fingerprint(o); if (S.lastFp[k] === f) return false; S.lastFp[k] = f; return true; };
      renderTicking(d);
      if (fp("queue", d.queue)) renderQueue(d);
      if (fp("budget", [d.budget, d.pods.map((p) => p.cost_usd)])) renderBudget(d);
      if (fp("volume", d.volume)) renderVolume(d);
      if (fp("world", d.world)) renderWorld(d);
      if (fp("events", [d.events, d.actions])) renderEvents(d);
      renderTop(d);
      app.setFarmStatus && app.setFarmStatus(statusItems(d));
      if (app.noteMonitor) app.noteMonitor(d.monitor);
    }

    function renderTicking(d) { renderPods(d); renderJobs(d); }

    function renderHead(d) {
      const sel = d.dances.map((x) => h("option", { value: x.id, selected: x.id === d.dance }, x.label + (x.complete_pct != null ? `  ·  ${Math.round(x.complete_pct)} % complete` : "")));
      replace(danceSel, sel);
      danceSel.value = d.dance;
      const w = d.world || {};
      replace(worldEl, w.id ? [h("span", { class: "chip mono", title: `world ${w.id}; the 3D poses are checked on the pose id ${w.pose_id || "?"}` }, `world ${w.id}`),
        w.sealed ? pill("sealed", "good", "the room model is solved and frozen: every analysis uses it") : pill("not sealed", "warn")] : null);
      const cov = d.coverage;
      const stale = d.outputs && d.outputs.stale;
      const refreshing = (d.actions || []).some((a) => a.label === "coverage" && (a.state === "running" || a.state === "starting"));
      replace(updatedEl, cov.exists ? [`map computed ${cov.updated_ts ? fmtAgo(cov.updated_ts, srvNow()) : cov.updated || "?"}`,
        refreshing ? pill("recomputing…", "info") : stale ? pill("new output: map refreshes", "info") : null,
        h("button", { class: "btn small ghost", title: "Recompute the map from the real outputs now", onclick: () => act("/api/farm/coverage/refresh", "Coverage refresh") }, "recompute")] : "no coverage yet");
      const cmd = d.commands || {};
      const wch = d.watcher || {};
      replace(watcherEl, wch.running ? pill(`watcher alive${wch.cycle != null ? " · cycle " + wch.cycle : ""}`, "good", "the detached watcher pulls finished chunks and keeps pods healthy")
        : wch.hung ? pill("watcher hung", "crit", "its process exists but its heartbeat is old: restart it")
          : wch.known ? pill(`watcher ${wch.state || "stopped"}`, "neutral") : pill("no watcher", "neutral", "it starts with Fill gaps (farm up)"));
      const busy = (label) => (d.actions || []).some((a) => a.label === label && (a.state === "running" || a.state === "starting"));
      bFill.disabled = !cov.exists || busy("fill");
      bPull.disabled = !cmd.pull || busy("pull");
      bPull.title = cmd.pull ? "Download every finished chunk from the volume now" : "Not installed yet: `farm pull` arrives with the watcher stage";
      bLog.title = d.logs && d.logs.length ? "The farm logs (watcher, actions)" : "No log yet: the watcher writes farm/watcher.log once it runs";
      bRestart.disabled = !cmd.watch;
      bRestart.title = cmd.watch ? "Stop the detached watcher and start it again" : "Not installed yet: `farm watch` arrives with the watcher stage";
      bStop.disabled = !d.pods.some((p) => p.stoppable);
      bStop.title = bStop.disabled ? "No farm pod is running (fit_full pods are stopped through fit_full, not here)"
        : "Pull what is finished, then delete this farm's pods (asks first; pods with unpulled results are kept)";
      bPng.disabled = !cov.exists;
    }

    function renderAlerts(d) {
      const al = d.alerts || [];
      replace(alertsEl, al.slice(0, 4).map((a) => h("div", { class: ["alert", "alert-" + a.level], role: a.level === "red" ? "alert" : null },
        h("span", { class: "alert-icon", "aria-hidden": "true" }, { red: "✕", amber: "!", info: "i" }[a.level] || "i"),
        h("span", { class: "alert-msg" }, a.id === "budget" ? h("strong", null, "Budget: ") : null, a.message))));
      alertsEl.hidden = !al.length;
    }

    function renderHero(d) {
      const cov = d.coverage;
      if (!cov.exists) {
        replace(heroEl, h("div", { class: "fm-empty" }, h("strong", null, "No coverage map yet."), " The map is computed from the real outputs: ",
          h("code", null, "python -m dancecap.farm --dance " + d.dance + " coverage")));
        return;
      }
      const pc = cov.percent, sec = cov.seconds, cn = cov.counts;
      const cells = cov.cells;
      const pb = { front: 0, fit: 0, refined: 0, other: 0 };
      for (const c of cells) if (c.s === "partial") pb[c.g === "front" ? "front" : c.g === "fit" ? "fit" : c.g === "refined" ? "refined" : "other"]++;
      const stat = (cls, big, label, secs, sub) => h("div", { class: ["fm-stat", cls] },
        h("div", { class: "fm-big" }, big, h("span", { class: "fm-pct" }, " %")), h("div", { class: "fm-statl" }, h("strong", null, label), h("span", { class: "muted" }, ` ${Math.round(secs)} s`)),
        sub ? h("div", { class: "fm-sub muted" }, sub) : null);
      const total = Math.max(1, (sec.complete || 0) + (sec.partial || 0) + (sec.none || 0));
      replace(heroEl,
        h("div", { class: "fm-stats" },
          stat("complete", fmtPct(pc.complete), "complete", sec.complete, `${cn.complete} cells: refined, stitched, gates passed`),
          stat("partial", fmtPct(pc.partial), "partial", sec.partial, `${pb.front} front only · ${pb.fit} raw fit · ${pb.refined} refined, awaiting stitch${pb.other ? ` · ${pb.other} other` : ""}`),
          stat("none", fmtPct(pc.none), "not done", sec.none, `${cn.none} cells with nothing usable`),
          h("div", { class: "fm-total" }, h("div", { class: "fm-total-n" }, fmtDur(cov.total_s || total)), h("div", { class: "muted small" }, `of final result · ${cov.n_cells} cells`))),
        h("div", { class: "fm-stack", role: "img", "aria-label": `${pc.complete}% complete, ${pc.partial}% partial, ${pc.none}% not done` },
          h("span", { class: "fm-seg complete", style: { width: `${100 * (sec.complete || 0) / total}%` } }),
          h("span", { class: "fm-seg partial", style: { width: `${100 * (sec.partial || 0) / total}%` } }),
          h("span", { class: "fm-seg none", style: { width: `${100 * (sec.none || 0) / total}%` } })));
    }

    function renderLegends(d) {
      const sw = (cls, text, title) => h("span", { class: "fm-lg", title }, h("i", { class: ["fm-sw", cls] }), text);
      replace(legendEl, sw("complete", "complete"), sw("hatch1", "front only"), sw("hatch2", "raw fit"), sw("hatch3", "refined, awaiting stitch"),
        sw("none", "not done"), sw("live", "being worked on now"), sw("mark", "mono / two-view / absent"));
      replace(chunkLegend, d.chunks.map((c, k) => h("span", { class: ["fm-chunk", c.status === "running" ? "run" : ""],
        title: `chunk ${c.id}: ${c.status}${c.pod ? " on pod " + c.pod : ""}${c.source === "fit_full" ? " (fit_full session)" : ""}` },
      h("i", { class: "fm-chunk-sw", style: { background: chunkColor(k) } }), h("strong", null, c.id), " ", fmtSpan(c.t0, c.t1), " · ", c.core_s != null ? `${Math.round(c.core_s)} s` : "", " · ",
      h("span", { class: "fm-chunk-st fm-chunk-" + c.status }, c.status))),
      !d.chunks.length ? h("span", { class: "muted small" }, "No chunks planned yet. The cells above come from the front-end and fit outputs on disk.") : null);
    }

    function renderTop(d) {
      const open = d.pods.filter((p) => p.state === "running" || p.state === "starting");
      const rate = open.reduce((a, p) => a + (p.price_hr || 0), 0);
      const b = d.budget;
      app.setTop && app.setTop(
        open.length ? pill(`${open.length} pod${open.length > 1 ? "s" : ""} · ${fmtUSD(rate)}/h`, "info", "pods billing now") : h("span", { class: "muted small" }, "no pods"),
        b && b.spent != null ? h("span", { class: ["fm-topbud", b.level === "red" ? "crit-text" : b.level === "amber" ? "warn-text" : ""], title: "spent / cap (ledger)" },
          `${fmtUSD(b.spent, 1)} / ${fmtUSD(b.cap, 0)}`) : null);
    }

    function statusItems(d) {
      return [h("span", { class: "sb-item" }, `${d.live.fit_full ? d.live.fit_full + " fit_full session(s) · " : ""}${d.chunks.filter((c) => c.status === "running").length} chunk(s) running`),
        h("span", { class: "sb-item", title: "heavy CV / neural work runs on Runpod; this PC only orchestrates" }, "compute: heavy → Runpod"),
        h("span", { class: "sb-item" }, d.watcher.running ? "watcher alive" : "watcher not running")];
    }

    // ---- panels
    function card(el, title, right, ...kids) {
      replace(el, h("div", { class: "fm-card-head" }, h("h3", null, title), h("span", { class: "spacer" }), right || null), ...kids);
    }

    function renderPods(d) {
      const pods = d.pods;
      const open = pods.filter((p) => p.state === "running" || p.state === "starting");
      const rate = open.reduce((a, p) => a + (p.price_hr || 0), 0), spend = pods.reduce((a, p) => a + (p.cost_usd || 0), 0);
      card(panels.pods, "Pods", h("span", { class: "muted small" }, pods.length ? `${open.length} open · ${fmtUSD(rate)}/h · ${fmtUSD(spend)} so far` : ""),
        pods.length ? h("table", { class: "fm-tbl" },
          h("thead", null, h("tr", null, ["GPU", "pod", "state", "$/h", "age", "spend", "chunks"].map((x) => h("th", null, x)))),
          h("tbody", null, pods.map((p) => h("tr", { title: p.note || "" },
            h("td", null, h("strong", null, GPU_SHORT(p.gpu)), h("div", null, srcPill(p.source))),
            h("td", { class: "mono" }, p.name ? `${p.name}` : "", h("div", { class: "muted small" }, (p.id || "").slice(0, 10))),
            h("td", null, pill(p.state || "?", p.state === "running" ? "good" : p.state === "starting" ? "info" : p.state === "lost" || p.state === "stuck" ? "crit" : p.state === "idle" ? "warn" : "neutral"),
              p.kind && p.kind !== "gpu" ? pill(p.kind, "neutral") : null, (p.flags || []).map((f) => pill(f, "crit"))),
            h("td", { class: "num" }, p.price_hr != null ? fmtUSD(p.price_hr) : "–"),
            h("td", { class: "num" }, p.created_ts ? fmtDur(srvNow() - p.created_ts, { short: true }) : p.age_s != null ? fmtDur(p.age_s, { short: true }) : "–"),
            h("td", { class: "num" }, p.cost_usd != null ? fmtUSD(p.cost_usd) : "–"),
            h("td", null, (p.chunks || []).join(", ") || "–"))))) : h("p", { class: "muted" }, "No pods are running. Fill gaps starts them (you confirm the cost first)."));
    }

    function jobRow(j, d) {
      const ci = d.chunks.findIndex((c) => c.id === j.chunk);
      const eta = etaText(j.eta);
      const pr = j.progress && j.progress.frac != null ? j.progress.frac : null;
      const hb = j.heartbeat_ts ? Math.max(0, Math.round(srvNow() - j.heartbeat_ts)) : j.heartbeat_age_s;
      return h("div", { class: ["fm-job", "fm-job-" + j.status] },
        h("div", { class: "fm-job-top" },
          h("i", { class: "fm-chunk-sw", style: { background: ci >= 0 ? chunkColor(ci) : "var(--text-3)" } }),
          h("strong", { class: "fm-job-id" }, j.id), h("span", { class: "muted" }, `${fmtSpan(j.span[0], j.span[1])}${j.core_s != null ? ` · ${Math.round(j.core_s)} s` : ""}`),
          pill(j.status, j.status === "running" ? "info" : j.status === "failed" ? "crit" : "neutral"), srcPill(j.source),
          h("span", { class: "spacer" }), j.step ? h("span", { class: "chip mono", title: "current step" }, j.step) : null),
        h("div", { class: "fm-job-mid" }, stepChips(j.steps), bar(pr, { running: j.status === "running", title: j.progress ? j.progress.basis : "no progress figure yet" }),
          h("span", { class: "num fm-job-pct", title: j.progress ? j.progress.basis : "" }, pr != null ? `${Math.round(pr * 100)} %` : "…")),
        h("div", { class: "fm-job-bot muted small" },
          j.status === "done" ? h("span", { class: "good-text" }, "finished: downloaded and checked")
            : eta ? h("span", { class: "fm-eta", title: eta.title }, eta.text) : h("span", null, "ETA not known yet"),
          h("span", { class: j.heartbeat_stale && j.status === "running" ? "warn-text" : "", title: "last sign of life from the pod" },
            hb != null ? (j.status === "running" ? `heartbeat ${hb < 90 ? hb + " s ago" : fmtDur(hb) + " ago (stale)"}` : `last heartbeat ${fmtDur(hb)} ago`)
              : "no heartbeat yet"),
          j.pod ? h("span", null, `${GPU_SHORT(j.gpu)} · ${j.pod.slice(0, 8)}`) : null,
          j.cost_usd != null ? h("span", null, `${fmtUSD(j.cost_usd)} so far`) : null,
          j.kill_at ? h("span", { title: "the on-pod kill switch (a safety net: extended while the fit is healthy)" }, `kill switch ${fmtTime(j.kill_at)}`) : null,
          j.error ? h("span", { class: "crit-text" }, j.error) : null));
    }

    function renderJobs(d) {
      const jobs = d.jobs;
      card(panels.jobs, "Chunk jobs", h("span", { class: "muted small" }, jobs.length ? `${jobs.filter((j) => j.status === "running").length} running` : ""),
        jobs.length ? jobs.map((j) => jobRow(j, d)) : h("p", { class: "muted" }, "Nothing is running. The queue below holds what is planned."));
    }

    function renderQueue(d) {
      card(panels.queue, "Queue", h("span", { class: "muted small" }, d.queue.length ? `${d.queue.length} waiting` : ""),
        d.queue.length ? h("table", { class: "fm-tbl" }, h("thead", null, h("tr", null, ["job", "span", "kind", "est.", "after"].map((x) => h("th", null, x)))),
          h("tbody", null, d.queue.map((q) => h("tr", null, h("td", null, h("strong", null, q.id), " ", pill(q.status, "neutral")),
            h("td", { class: "num" }, q.span[0] != null ? fmtSpan(q.span[0], q.span[1]) : "–"), h("td", null, q.kind || "–"),
            h("td", { class: "num" }, fmtMin(q.est_min)), h("td", { class: "mono small" }, (q.after || []).join(", ") || "–"))))) : h("p", { class: "muted" }, "Queue is empty."));
    }

    function renderBudget(d) {
      const b = d.budget;
      const level = b.level || "green";
      const cap = b.cap, spent = b.spent;
      card(panels.budget, "Budget", h("span", { class: ["lvl", "lvl-" + level] }, { green: "within budget", amber: "watch it", red: "over" }[level] || level),
        spent != null && cap ? [h("div", { class: "fm-money" }, h("strong", null, fmtUSD(spent)), h("span", { class: "muted" }, ` spent of ${fmtUSD(cap, 0)} cap`)),
          h("div", { class: "fm-capbar", title: "spent / cap" }, h("div", { style: { width: `${clamp(100 * spent / cap, 0, 100)}%` } }))] : h("p", { class: "muted" }, b.connection ? "Runpod ledger not read yet." : "Waiting for the Runpod ledger…"),
        h("div", { class: "kv fm-kv" },
          h("span", null, "account balance"), h("span", null, b.balance != null ? fmtUSD(b.balance) : "–"),
          h("span", null, "cap left"), h("span", null, b.remaining != null ? fmtUSD(b.remaining) : "–"),
          h("span", { title: "min(cap left, balance - $2) minus the worst case of pods already open" }, "free for new pods"), h("span", { class: b.available_usd != null && b.available_usd < 2 ? "warn-text" : "" }, b.available_usd != null ? fmtUSD(b.available_usd) : "–"),
          h("span", null, "billing now"), h("span", null, b.open_rate_hr != null ? `${fmtUSD(b.open_rate_hr)}/h` : "–")),
        (b.messages || []).map((m) => h("p", { class: "warn-text small" }, m)));
    }

    function renderVolume(d) {
      const v = d.volume;
      if (!v) { card(panels.volume, "Persistent volume", null, h("p", { class: "muted" }, "No volume yet (farm_volume.json).")); return; }
      const frac = v.used_gb != null && v.size_gb ? v.used_gb / v.size_gb : null;
      card(panels.volume, "Persistent volume", pill(v.dc || "?", "neutral", `${v.tier || ""} storage`),
        h("div", { class: "fm-money" }, h("strong", null, v.used_gb != null ? `${v.used_gb.toFixed(1)} GB` : "?"), h("span", { class: "muted" }, ` used of ${v.size_gb} GB`)),
        h("div", { class: "fm-capbar", title: "used / size" }, h("div", { style: { width: `${frac != null ? clamp(frac * 100, 0.5, 100) : 0}%` } })),
        h("div", { class: "kv fm-kv" },
          h("span", null, "name · id"), h("span", { class: "mono" }, `${v.name} · ${v.id}`),
          h("span", null, "price"), h("span", null, v.price_month != null ? `${fmtUSD(v.price_month)}/month (${v.size_gb} GB × ${fmtUSD(v.price_gb_month)})` : "–"),
          h("span", null, "usage measured"), h("span", { class: "muted" }, v.used_gb != null ? `${v.used_measured || "?"} (${v.used_source || "?"})` : "not measured: no pod mounts it now")));
    }

    function renderWorld(d) {
      const w = d.world || {};
      card(panels.world, "World package", w.sealed != null ? pill(w.sealed ? "sealed" : "not sealed", w.sealed ? "good" : "warn") : null,
        w.id ? h("div", { class: "kv fm-kv" }, h("span", null, "world id"), h("span", { class: "mono" }, w.id),
          h("span", null, "pose id"), h("span", { class: "mono" }, w.pose_id || "–"), h("span", null, "avatar id"), h("span", { class: "mono" }, w.avatar_id || "–"))
          : h("p", { class: "muted" }, "No world is pinned to this dance."),
        h("p", { class: "muted small" }, "Every chunk is solved against this one room model, skeleton and avatar set. The stitcher refuses poses with another pose id."));
    }

    function renderEvents(d) {
      const acts = (d.actions || []).slice(0, 4);
      card(panels.events, "Recent activity", null,
        acts.length ? h("div", { class: "fm-acts" }, acts.map((a) => h("div", { class: "fm-act" }, pill(a.state, a.state === "ok" ? "good" : a.state === "failed" || a.state === "lost" ? "crit" : "info"),
          h("strong", null, a.label), h("span", { class: "muted small" }, a.started ? fmtAgo(a.started, d.now) : ""),
          a.error ? h("span", { class: "crit-text small" }, a.error) : null,
          a.log ? h("button", { class: "btn small ghost", onclick: () => openLog(a.log) }, "log") : null))) : null,
        d.events.length ? h("table", { class: "fm-tbl" }, h("tbody", null, d.events.slice(0, 8).map((e) => h("tr", null,
          h("td", { class: "muted num" }, (e.ts || "").slice(11, 19)), h("td", null, h("span", { class: "chip" }, e.kind || "")), h("td", null, e.msg)))))
          : h("p", { class: "muted" }, "No farm events yet."));
    }

    // ---- actions
    async function act(url, label, extra = {}) {
      try {
        const res = await api.post(url, { dance: S.dance, ...extra });
        toast(`${label}: started${res && res.id ? " (" + res.id.slice(-8) + ")" : ""}.`, "good");
        await poll(true);
        return res;
      } catch (e) {
        toast(`${label}: ${errorText(e)}`, e.status === 409 ? "warn" : "crit");
        return null;
      }
    }

    async function fillGaps(podsWanted) {
      const d = S.data;
      let plan;
      const waiting = openDialog({ title: "Fill gaps: planning…", body: h("div", { class: "confirm-msg" }, h("p", null, "Asking the planner which spans still need 3D poses, and what the pods cost now. Nothing is started yet."),
        bar(null)), actions: [{ label: "Cancel" }] });
      try {
        plan = await api.post("/api/farm/fill/plan", { dance: d.dance, ...(podsWanted ? { pods: podsWanted } : {}) });
      } catch (e) { waiting.close(); toast(`Fill gaps: ${errorText(e)}`, "crit"); return; }
      waiting.close();
      showPlan(plan);
    }

    function showPlan(plan) {
      const est = plan.estimate, b = plan.budget;
      const podsSel = h("select", { "aria-label": "Number of pods", onchange: () => { dlg.close(); fillGaps(Number(podsSel.value)); } },
        [1, 2, 3, 4, 5, 6].map((n) => h("option", { value: n, selected: n === est.pods }, `${n} pod${n > 1 ? "s" : ""}`)));
      const rows = plan.chunks.map((c) => h("tr", null, h("td", null, h("strong", null, c.id)), h("td", { class: "num" }, c.core_s != null ? `${Math.round(c.core_s)} s` : "–"),
        h("td", null, c.kind), h("td", { class: "num" }, fmtMin(c.est_min)),
        h("td", null, (est.assignment.find((a) => a.chunks.includes(c.id)) || {}).pod ? `pod ${(est.assignment.find((a) => a.chunks.includes(c.id))).pod}` : "–")));
      const gate = b.gate;
      const cpuOnly = !plan.chunks.length && plan.final_cpu;
      const body = plan.nothing_to_do ? h("div", { class: "confirm-msg" }, h("p", null, plan.message),
        plan.excluded.length ? h("ul", null, plan.excluded.map((x) => h("li", null, h("strong", null, x.id), ` – ${x.why}`))) : null)
        : h("div", { class: "confirm-msg fm-plan" },
          h("p", null, cpuOnly
            ? "Every chunk has finished. This plans the last step: stitch the chunks into one result, run timing, physics, counterbalance and floorcraft on it, and write the coverage evidence, on one CPU pod (no GPU)."
            : `${plan.chunks.length} chunk${plan.chunks.length > 1 ? "s" : ""} of about 30 s, solved against world ${(S.data.world || {}).id || "?"} (the same room model, skeleton and avatars), then stitched. Finished chunks are downloaded as soon as they are done.`),
          plan.chunks.length ? h("table", { class: "fm-tbl" }, h("thead", null, h("tr", null, ["chunk", "core", "kind", "est. time", "runs on"].map((x) => h("th", null, x)))), h("tbody", null, rows)) : null,
          plan.excluded.length ? h("p", { class: "muted small" }, "Not run here: ", plan.excluded.map((x) => `${x.id} (${x.why})`).join("; "), ".",
            plan.live_fit_full.length ? " Chunks that fit_full is fitting are marked as running so the queue skips them." : "") : null,
          plan.final_blocked ? h("p", { class: "muted small" }, "The stitch and analysis job is planned once the fit_full chunks have finished and been pulled.") : null,
          h("div", { class: "fm-plan-grid" },
            h("div", null, h("div", { class: "flabel" }, "Pods"), cpuOnly ? h("div", null, "1 CPU pod, about $0.28/h (8 vCPU)")
              : h("div", null, podsSel, ` × ${GPU_SHORT(plan.gpu.gpu)} at ${fmtUSD(plan.gpu.price)}/h`),
              est.price_note && !cpuOnly ? h("div", { class: "warn-text small" }, est.price_note) : null),
            h("div", null, h("div", { class: "flabel" }, "Time"), h("div", null, `about ${fmtDur(est.wall_min * 60)}`),
              h("div", { class: "muted small" }, cpuOnly ? "pod start + stitch + analyses" : `longest pod ${fmtDur(est.makespan_min * 60)} + stitch and analysis`)),
            h("div", null, h("div", { class: "flabel" }, "Cost"), h("div", null, h("strong", null, `about ${fmtUSD(est.expected_cost_usd)}`), `, worst case ${fmtUSD(est.worst_cost_usd)}`),
              h("div", { class: "muted small" }, "worst case = every pod runs to its kill switch")),
            h("div", null, h("div", { class: "flabel" }, "Budget gate"), h("div", { class: gate === "over" ? "crit-text" : gate === "ok" ? "good-text" : "warn-text" },
              gate === "ok" ? `ok (free ${fmtUSD(b.available_usd)})` : gate === "over" ? `over: only ${fmtUSD(b.available_usd)} free` : "balance unknown: the launcher checks it again"))),
          cpuOnly ? h("p", { class: "muted small" }, "Chunks that were fitted outside the farm are uploaded to the volume first (their refined motion; the home link is slow). Time and cost are estimates.")
            : h("p", { class: "muted small" }, `Time and cost are estimates from the S1 / 38 s reference runs on an A100 (about 1.35 min of pod time per second of window)${plan.slowdown && plan.slowdown !== 1 ? `, times ${plan.slowdown} for this GPU: a full fit on it is not measured yet` : ""}.`),
          plan.missing_commands.length ? h("p", { class: "warn-text small" }, `Not installed yet: ${plan.missing_commands.join(", ")}. The steps that need them are skipped or the run stops with a clear message.`) : null,
          h("details", null, h("summary", { class: "small muted" }, "commands that will run"), h("pre", { class: "fm-pre" }, plan.commands.join("\n"))));
      const actions = [{ label: "Cancel" }];
      if (!plan.nothing_to_do) {
        actions.push({ label: `${cpuOnly ? "Stitch and analyse" : "Fill gaps"}: spend up to ${fmtUSD(est.worst_cost_usd)}`, kind: "primary", disabled: gate === "over", onClick: async () => {
          const what = cpuOnly ? "1 CPU pod (the stitch and analysis job)" : `${est.pods} pod${est.pods > 1 ? "s" : ""} (${GPU_SHORT(plan.gpu.gpu)}, ${fmtUSD(plan.gpu.price)}/h each)`;
          const ok = await confirmDialog("Start pods?", `This starts ${what}. The worst case is ${fmtUSD(est.worst_cost_usd)}; the expected cost is about ${fmtUSD(est.expected_cost_usd)}. Every pod has a kill switch.`, { okLabel: "Start pods" });
          if (!ok) return false;
          try {
            await api.post("/api/farm/fill/run", { dance: plan.dance, plan_hash: plan.plan_hash, approve: { max_cost_usd: est.worst_cost_usd } });
            toast("Fill gaps is running. Follow it under Recent activity.", "good");
            poll(true);
          } catch (e) { toast(`Fill gaps: ${errorText(e)}`, "crit"); return false; }
        } });
      }
      const dlg = openDialog({ title: `Fill gaps: ${plan.dance}`, body, actions, wide: true });
    }

    async function openLog(name) {
      let cur = name || null;
      const pre = h("pre", { class: "fm-pre fm-log" }, "loading…");
      const sel = h("select", { "aria-label": "log", onchange: () => { cur = sel.value; load(); } });
      const info = h("span", { class: "muted small" });
      async function load() {
        try {
          const r = await api.get(`/api/farm/log?dance=${encodeURIComponent(S.dance)}${cur ? "&name=" + encodeURIComponent(cur) : ""}`);
          cur = r.name;
          replace(sel, (r.names || []).map((n) => h("option", { value: n, selected: n === r.name }, n)));
          replace(info, r.name ? `${r.path} · ${r.mtime ? fmtAgo(r.mtime) : ""}` : r.message);
          pre.textContent = r.text || r.message || "(empty)";
          pre.scrollTop = pre.scrollHeight;
        } catch (e) { pre.textContent = errorText(e); }
      }
      openDialog({ title: "Farm logs", wide: true, body: h("div", { class: "confirm-msg" }, h("div", { class: "row" }, sel, info), pre),
        actions: [{ label: "Reveal in Explorer", onClick: async () => { try { await api.post("/api/farm/reveal", { dance: S.dance, what: "logs" }); } catch (e) { toast(errorText(e), "warn"); } return false; } },
          { label: "Refresh", onClick: async () => { await load(); return false; } }, { label: "Close", kind: "primary" }] });
      load();
    }

    async function restartWatcher() {
      const ok = await confirmDialog("Restart the watcher?", "The watcher pulls finished chunks, keeps the pods' kill switches extended while they are healthy and deletes a pod only after its chunks are downloaded and verified. Restarting stops the running watcher and starts a new detached one; nothing it has already pulled is lost.", { okLabel: "Restart watcher" });
      if (ok) await act("/api/farm/watcher/restart", "Restart watcher");
    }

    async function stopPods() {
      const farm = S.data.pods.filter((p) => p.stoppable);
      if (!farm.length) return;
      const undone = S.data.chunks.filter((c) => c.source === "farm" && ["queued", "running", "done"].includes(c.status));
      const typed = await typedConfirm("Stop pods", h("div", { class: "confirm-msg" },
        h("p", null, `Deletes ${farm.length} farm pod${farm.length > 1 ? "s" : ""}: ${farm.map((p) => `${p.name || p.id} (${GPU_SHORT(p.gpu)})`).join(", ")}.`),
        undone.length ? h("p", { class: "warn-text" }, `Chunks ${undone.map((c) => c.id).join(", ")} are queued, running or done but not pulled yet: whatever a deleted pod has not saved to the volume is lost.`) : null,
        h("p", { class: "muted small" }, "fit_full pods and pods that are not this farm's are never touched from here.")), "STOP", { okLabel: "Delete pods" });
      if (typed) await act("/api/farm/pods/stop", "Stop pods", { pods: farm.map((p) => p.id), confirm: typed });
    }

    async function exportPng() {
      const d = S.data, cov = d.coverage;
      const stat = (big, label, sub, color) => ({ big: `${big} %`, label, sub, color });
      const css = getComputedStyle(canvas);
      const colors = { c: css.getPropertyValue("--f-complete").trim() || "#2ea043", p: css.getPropertyValue("--f-partial").trim() || "#f0a81c", n: css.getPropertyValue("--text-3").trim() || "#999" };
      const blob = await map.exportBlob({
        title: `${d.dance}  3D poses`, sub: `world ${(d.world || {}).id || "?"}${(d.world || {}).sealed ? " (sealed)" : ""}   ${Math.round(cov.total_s)} s (${mmss(cov.total_s)})   map computed ${cov.updated || "?"}`,
        stats: [stat(fmtPct(cov.percent.complete), "complete", `${Math.round(cov.seconds.complete)} s`, colors.c), stat(fmtPct(cov.percent.partial), "partial", `${Math.round(cov.seconds.partial)} s`, colors.p),
          stat(fmtPct(cov.percent.none), "not done", `${Math.round(cov.seconds.none)} s`, colors.n)],
        legend: d.chunks.map((c, k) => ({ color: chunkColor(k), text: `${c.id}  ${fmtSpan(c.t0, c.t1)}  ${c.core_s != null ? Math.round(c.core_s) + " s" : ""}  ${c.status}${c.pod ? "  pod " + c.pod.slice(0, 6) : ""}` })),
        foot: `exported ${new Date().toLocaleString()} from the Pipeline Monitor`,
      });
      const b64 = await new Promise((res) => { const r = new FileReader(); r.onload = () => res(String(r.result).split(",")[1]); r.readAsDataURL(blob); });
      try {
        const r = await api.post("/api/farm/map-png", { dance: d.dance, png_b64: b64 });
        toast(`Map saved: ${r.path}`, "good", { timeout: 9000, action: { label: "Reveal", onClick: () => api.post("/api/farm/reveal", { dance: d.dance, what: "map" }).catch(() => {}) } });
      } catch (e) { toast(`Export failed: ${errorText(e)}`, "crit"); }
    }

    // ---- navigation and polling
    function go(dance) {
      S.dance = dance; S.selected = -1; S.lastFp = {}; S.data = null; S.seq++;
      const target = `#/farm/${encodeURIComponent(dance)}`;
      if (location.hash !== target) history.replaceState(null, "", target);
      poll(true);
    }

    async function poll(force) {
      if (S.destroyed || (S.etagBusy && !force)) return;
      S.etagBusy = true;
      const seq = ++S.seq;
      try {
        const q = S.dance ? `?dance=${encodeURIComponent(S.dance)}` : "";
        const { data, changed } = await api.getx(`/api/farm${q}`);
        if (seq !== S.seq) return;   // another dance was chosen meanwhile: this answer is stale
        S.error = null; S.lastOk = Date.now();
        if (changed || !S.data) { S.fetchedAt = Date.now(); renderAll(data); }
        else { renderTicking(S.data); renderHead(S.data); if (app.noteMonitor) app.noteMonitor(data.monitor); }
      } catch (e) {
        if (seq !== S.seq) return;
        if (e && e.status === 404 && S.dance && !route.dance) {   // the remembered dance is gone: fall back to the newest
          S.dance = null; prefs.set("farm_dance", null); S.etagBusy = false; return poll(true);
        }
        S.error = e instanceof TypeError ? "offline" : errorText(e);
        if (!S.data) replace(heroEl, h("div", { class: "fm-empty" }, h("strong", null, "The monitor cannot read the farm: "), S.error));
        else toast(`Farm view: ${S.error}`, "warn", { timeout: 3000 });
        if (api.isAuthError && api.isAuthError(e)) app.authExpired && app.authExpired();
      } finally { S.etagBusy = false; }
    }

    function schedule() {
      clearTimeout(S.timer);
      if (S.destroyed) return;
      S.timer = setTimeout(async () => { await poll(); schedule(); }, document.visibilityState === "visible" ? POLL_VISIBLE_MS : POLL_HIDDEN_MS);
    }
    const onVis = () => { if (document.visibilityState === "visible") { poll(); schedule(); } };
    document.addEventListener("visibilitychange", onVis);

    await poll(true);
    schedule();
    return {
      route(r) { if (r.dance && r.dance !== S.dance) go(r.dance); },
      destroy() { S.destroyed = true; clearTimeout(S.timer); document.removeEventListener("visibilitychange", onVis); map.destroy(); tip.remove(); app.setFarmStatus && app.setFarmStatus(null); },
      refresh() { /* the view polls /api/farm itself */ },
    };
  },
};
