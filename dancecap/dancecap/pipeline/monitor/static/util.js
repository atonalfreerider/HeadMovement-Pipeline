// DOM builders and formatters. Every server string is untrusted text: it only ever reaches the page through
// textContent / attribute values (never innerHTML), per PIPELINE_MONITOR.md 7.6.

const SVG_NS = "http://www.w3.org/2000/svg";

function append(el, kids) {
  for (const k of kids) {
    if (k === null || k === undefined || k === false || k === true) continue;
    if (Array.isArray(k)) { append(el, k); continue; }
    el.appendChild(k instanceof Node ? k : document.createTextNode(String(k)));
  }
}

function setAttrs(el, attrs, isSvg) {
  if (!attrs) return;
  for (const [k, v] of Object.entries(attrs)) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.setAttribute("class", Array.isArray(v) ? v.filter(Boolean).join(" ") : v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (!isSvg && (k === "value" || k === "checked" || k === "disabled" || k === "selected")) el[k] = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
}

/** h("div", {class: "x", onclick: fn}, "text", child, [more]) */
export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  setAttrs(el, attrs, false);
  append(el, kids);
  return el;
}

export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  setAttrs(el, attrs, true);
  append(el, kids);
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function replace(el, ...kids) {
  clear(el);
  append(el, kids);
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ------------------------------------------------------------------------------------------------ formatting

export function fmtDur(sec, opts = {}) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "–";
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return opts.short ? `${sec}s` : `${sec} s`;
  const m = Math.round(sec / 60);
  if (m < 60) return opts.short ? `${m}m` : `${m} min`;
  const hh = Math.floor(sec / 3600), mm = Math.round((sec - hh * 3600) / 60);
  if (mm === 60) return `${hh + 1} h 00`;
  return opts.short ? `${hh}h${String(mm).padStart(2, "0")}` : `${hh} h ${String(mm).padStart(2, "0")}`;
}

/** "14 min (p90 22)" */
export function fmtEta(e) {
  if (!e) return "–";
  const p50 = e.p50_s ?? e.p50, p90 = e.p90_s ?? e.p90;
  if (p50 === undefined || p50 === null) return "–";
  if (p90 === undefined || p90 === null || Math.abs(p90 - p50) < 30) return fmtDur(p50);
  return `${fmtDur(p50)} (p90 ${fmtDur(p90, { short: true })})`;
}

export function fmtUSD(x, digits = 2) {
  if (x === null || x === undefined || !isFinite(x)) return "–";
  return (x < 0 ? "-$" : "$") + Math.abs(x).toFixed(digits);
}

export function fmtCostBand(c) {
  if (!c) return "–";
  const a = c.p50 ?? c.p50_usd, b = c.p90 ?? c.p90_usd;
  if (a === undefined || a === null) return "–";
  if (!a && !b) return "$0";
  if (b === undefined || b === null || Math.abs(b - a) < 0.005) return fmtUSD(a);
  return `${fmtUSD(a)} (p90 ${fmtUSD(b)})`;
}

export function fmtBytes(b) {
  if (b === null || b === undefined || !isFinite(b)) return "–";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
  return `${b < 10 && i ? b.toFixed(1) : Math.round(b)} ${u[i]}`;
}

export function fmtTime(epoch, withDate = false) {
  if (!epoch) return "–";
  const d = new Date(epoch * 1000);
  const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false });
  if (!withDate) {
    const today = new Date();
    if (d.toDateString() === today.toDateString()) return t;
  }
  return `${d.toLocaleDateString([], { month: "short", day: "numeric" })} ${t}`;
}

export function fmtAgo(epoch, now = Date.now() / 1000) {
  if (!epoch) return "–";
  const d = now - epoch;
  if (d < 0) return "now";
  if (d < 60) return `${Math.round(d)} s ago`;
  if (d < 3600) return `${Math.round(d / 60)} min ago`;
  if (d < 86400) return `${Math.round(d / 3600)} h ago`;
  return `${Math.round(d / 86400)} d ago`;
}

export function fmtClock(sec) {
  if (sec === null || sec === undefined || !isFinite(sec)) return "–";
  const m = Math.floor(sec / 60), s2 = sec - m * 60;
  return `${m}:${s2.toFixed(1).padStart(4, "0")}`;
}

export function pct(f, digits = 0) {
  if (f === null || f === undefined || !isFinite(f)) return "–";
  return `${(f * 100).toFixed(digits)} %`;
}

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export function debounce(fn, ms) {
  let t = null;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/** Per-viewer conveniences only (remembered tab, collapsed sidebar). Never state that must persist. */
export const prefs = {
  get(k, dflt) {
    try { const v = localStorage.getItem("dcm." + k); return v === null ? dflt : JSON.parse(v); } catch { return dflt; }
  },
  set(k, v) {
    try { localStorage.setItem("dcm." + k, JSON.stringify(v)); } catch { /* storage blocked: fine */ }
  },
};

export const session = {
  get(k, dflt) {
    try { const v = sessionStorage.getItem("dcm." + k); return v === null ? dflt : JSON.parse(v); } catch { return dflt; }
  },
  set(k, v) {
    try { sessionStorage.setItem("dcm." + k, JSON.stringify(v)); } catch { /* ignore */ }
  },
  del(k) {
    try { sessionStorage.removeItem("dcm." + k); } catch { /* ignore */ }
  },
};

/** Stable fingerprint of an object minus volatile keys (so a re-render happens only on a real change). */
export function fingerprint(o, drop = ["computed_at", "now", "checked", "rev"]) {
  return JSON.stringify(o, (k, v) => (drop.includes(k) ? undefined : v));
}

export function esc(s2) { return String(s2 ?? ""); }

export function camLabel(cam) { return `cam ${cam}`; }

/** Only in-app hash routes are followed from server data (alert / gate / next-action links). */
export function safeRoute(link) {
  return typeof link === "string" && /^#\/[\w\-./]*$/.test(link) ? link : null;
}
