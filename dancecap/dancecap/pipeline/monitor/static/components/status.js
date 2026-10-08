// Stage status, "where it runs" badges, pills and the per-take status strip.
// Every status carries a glyph AND a colour (never colour alone): PIPELINE_MONITOR.md 6.1.
import { h } from "../util.js";

export const STATUS = {
  running: { label: "running", glyph: "▶", order: 0 },
  queued: { label: "queued", glyph: "◷", order: 1 },
  failed: { label: "failed", glyph: "✕", order: 2 },
  stopped: { label: "stopped", glyph: "■", order: 3 },
  blocked: { label: "needs you", glyph: "?", order: 4 },
  stale: { label: "stale", glyph: "↻", order: 5 },
  done: { label: "done", glyph: "✓", order: 6 },
  ready: { label: "ready", glyph: "○", order: 7 },
  waiting: { label: "waiting", glyph: "·", order: 8 },
  planned: { label: "not built yet", glyph: "◌", order: 9 },
  skipped: { label: "skipped", glyph: "–", order: 10 },
  "n/a": { label: "n/a", glyph: "–", order: 11 },
};

export function statusMeta(st) {
  return STATUS[st] || { label: st || "?", glyph: "?", order: 99 };
}

export function statusClass(st) {
  return "st-" + String(st || "unknown").replace("/", "");
}

export function statusBadge(st, extra) {
  const m = statusMeta(st);
  return h("span", { class: ["st", statusClass(st)], title: extra || m.label },
    h("span", { class: "st-glyph", "aria-hidden": "true" }, m.glyph), m.label);
}

export function statusGlyph(st) {
  const m = statusMeta(st);
  return h("span", { class: ["st-g", statusClass(st)], title: m.label, "aria-label": m.label }, m.glyph);
}

export const WHERE = {
  local: { badge: "L", label: "this PC (CPU)" },
  local_gpu: { badge: "G", label: "this PC's 2080 Ti (gpu.lock)" },
  pod: { badge: "P", label: "Runpod pod (community cloud)" },
  pod_secure: { badge: "S", label: "Runpod pod (secure cloud)" },
  human: { badge: "H", label: "you (a person)" },
  unity: { badge: "U", label: "Unity" },
};

export function whereBadge(runsOn) {
  const w = WHERE[runsOn] || { badge: "?", label: runsOn || "unknown" };
  return h("span", { class: ["where", "where-" + (runsOn || "x")], title: `runs on: ${w.label}` }, w.badge);
}

export function pill(text, kind = "neutral", title) {
  return h("span", { class: ["pill", "pill-" + kind], title }, text);
}

const CONN_KIND = {
  ok: "good", connected: "good", creating: "info", starting: "info", degraded: "warn", lost: "crit", ended: "neutral",
  no_key: "neutral", api_error: "crit", unknown: "neutral",
};
const CONN_LABEL = {
  ok: "Runpod connected", no_key: "Runpod: no API key", api_error: "Runpod API error", unknown: "Runpod: checking",
};

export function connPill(conn, short) {
  const label = short ? (conn || "unknown") : (CONN_LABEL[conn] || conn || "unknown");
  return pill([h("span", { class: "dot", "aria-hidden": "true" }), label], CONN_KIND[conn] || "neutral");
}

/** One cell per stage in DAG order. Shape differs per status (full / outline / short / hatched / dotted). */
export function strip(entries, { onClick } = {}) {
  const el = h("div", { class: "strip", role: "img",
    "aria-label": summarise(entries) });
  for (const [id, st] of entries || []) {
    if (st === "skipped" || st === "n/a") continue;
    el.appendChild(h("span", {
      class: ["cell", statusClass(st)], title: `${id}: ${statusMeta(st).label}`,
      onclick: onClick ? (e) => { e.stopPropagation(); onClick(id); } : null,
    }));
  }
  return el;
}

function summarise(entries) {
  const c = {};
  for (const [, st] of entries || []) c[st] = (c[st] || 0) + 1;
  return Object.entries(c).map(([k, v]) => `${v} ${statusMeta(k).label}`).join(", ");
}

export function legend(keys = ["done", "running", "queued", "stale", "failed", "blocked", "ready", "waiting", "planned"]) {
  return h("div", { class: "legend" }, keys.map((k) => statusBadge(k)));
}
