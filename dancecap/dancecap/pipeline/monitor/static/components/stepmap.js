// The analysis map: one row per step of an analysis run, one square per pod x 2 min of the step's plan (the defrag look of
// the Farm page, applied to the whole analysis). Same rule as snapshots.step_blocks (pipeline/snapshots.py), so the live
// page and the PNG snapshots agree. Also: the compact strip (one square per step) and the upload sparkline.
import { h, s, fmtDur, fmtUSD, clamp } from "../util.js";
import { pill } from "./status.js";

export const STATUS_TEXT = { waiting_agent: "needs an agent", pending: "pending", running: "running", done: "done",
  failed: "failed", blocked: "blocked", skipped: "skipped" };
export const STATUS_KIND = { done: "good", running: "info", waiting_agent: "warn", failed: "crit", blocked: "crit", pending: "neutral", skipped: "neutral" };

export function stepBlocks(st, now, unitMin = 2, cap = 90) {
  const lanes = st.where !== "local" ? (st.lanes || 1) : 1;
  const per = Math.max(1, Math.round((st.p50_min || 1) / unitMin));
  const n = Math.min(cap, lanes * per);
  const status = st.status || "pending";
  if (status === "done") return Array(n).fill("done");
  if (status === "running") {
    let frac = null;
    const pr = st.progress || {};
    if (pr.total) frac = (pr.done || 0) / pr.total;
    else if (st.started) frac = Math.min(0.95, (now - st.started) / 60 / Math.max(0.1, st.p50_min || 1));
    const k = Math.round(clamp(frac || 0, 0, 1) * n);
    return Array(k).fill("run").concat(Array(n - k).fill("left"));
  }
  const k = { waiting_agent: "wait", failed: "fail", blocked: "fail", skipped: "skip" }[status] || "n";
  return Array(n).fill(k);
}

export function blocks(kinds, title) {
  return h("div", { class: "am-sq", title }, kinds.map((k) => h("i", { class: ["am-b", k] })));
}

export function legend() {
  const sw = (k, t) => h("span", { class: "am-lg" }, h("i", { class: ["am-b", k] }), t);
  return h("span", { class: "am-legend" }, sw("done", "done"), sw("run", "running (filled = done so far)"), sw("wait", "needs an agent"),
    sw("n", "pending"), sw("fail", "failed"), h("span", { class: "muted" }, "one square = one pod × 2 min of the plan"));
}

export function stepLine(st, now) {
  const status = st.status || "pending";
  let dur;
  if (st.started && st.ended) dur = `took ${fmtDur(st.ended - st.started)}`;
  else if (st.started) dur = `running ${fmtDur(now - st.started)} of ~${Math.round(st.p50_min)} min`;
  else dur = `plan ${Math.round(st.p50_min)} min (p90 ${Math.round(st.p90_min)})`;
  if (status === "running" && st.eta_s != null) dur += ` · ETA ${fmtDur(st.eta_s)}`;
  return dur;
}

/** One square per step (the strip on the Farm page). */
export function strip(steps) {
  return h("span", { class: "am-strip" }, (steps || []).map((st) => h("i", {
    class: ["am-b", { done: "done", running: "run", waiting_agent: "wait", failed: "fail", blocked: "fail", skipped: "skip" }[st.status] || "n"],
    title: `${st.id}: ${STATUS_TEXT[st.status] || st.status}${st.where ? " · " + st.where : ""}` })));
}

/** Upload rate history [[t, MB], ...] as a small SVG (MB/s between samples). */
export function sparkline(hist, { w = 160, hh = 28 } = {}) {
  if (!hist || hist.length < 3) return null;
  const rates = [];
  for (let i = 1; i < hist.length; i++) {
    const dt = hist[i][0] - hist[i - 1][0];
    if (dt > 0) rates.push([hist[i][0], (hist[i][1] - hist[i - 1][1]) / dt]);
  }
  if (rates.length < 2) return null;
  const t0 = rates[0][0], t1 = rates[rates.length - 1][0], mx = Math.max(0.05, ...rates.map((r) => r[1]));
  const pts = rates.map((r) => `${((r[0] - t0) / Math.max(1, t1 - t0) * (w - 2) + 1).toFixed(1)},${(hh - 2 - r[1] / mx * (hh - 4)).toFixed(1)}`).join(" ");
  return s("svg", { class: "am-spark", width: String(w), height: String(hh), viewBox: `0 0 ${w} ${hh}`, role: "img",
    "aria-label": `upload rate, peak ${mx.toFixed(2)} MB/s` },
  s("polyline", { points: pts, fill: "none", stroke: "currentColor", "stroke-width": "1.5" }));
}

export function uploadText(up) {
  if (!up || !up.total) return null;
  const mb = (b) => (b / 1e6).toFixed(b >= 1e8 ? 0 : 1);
  if (up.done) return `${mb(up.total)} MB uploaded${up.mbps ? ` at ${up.mbps.toFixed(2)} MB/s` : ""}${up.streams > 1 ? ` (${up.streams} streams)` : ""}`;
  if (up.failed) return `upload failed after ${mb(up.sent || 0)} of ${mb(up.total)} MB`;
  if (up.sent == null) return `${mb(up.total)} MB to upload (${up.source || "waiting"})`;
  return `${mb(up.sent)}/${mb(up.total)} MB${up.mbps ? ` at ${up.mbps.toFixed(2)} MB/s` : ""}${up.eta_s != null ? `, ${fmtDur(up.eta_s)} left` : ""}${up.streams > 1 ? ` · ${up.streams} streams` : ""}`;
}

export function uploadBar(up) {
  if (!up || !up.total || up.done || up.sent == null) return null;
  const f = clamp(up.sent / up.total, 0, 1);
  return h("div", { class: "fm-bar running", role: "progressbar", "aria-valuenow": String(Math.round(f * 100)), "aria-valuemin": "0", "aria-valuemax": "100" },
    h("div", { class: "fm-bar-fill", style: { width: `${f * 100}%` } }));
}

export function sessionStatePill(c) {
  const st = c.state || "?";
  const kind = { running: "info", preparing: "info", ok: "good", failed: "crit", orphaned: "crit", ended: "neutral" }[st] || "neutral";
  const title = c.jobs_bad && Object.keys(c.jobs_bad).length ? Object.entries(c.jobs_bad).map(([k, v]) => `${k}: ${v}`).join("; ")
    : st === "orphaned" ? "the runner process is gone while the pod is open: nothing downloads its outputs" : null;
  return pill(st === "orphaned" ? "runner gone" : st, kind, title);
}

export function costText(c) {
  const cost = c.cost || {};
  const so = cost.so_far != null ? cost.so_far : c.cost_usd;
  if (cost.p50 != null) return `${fmtUSD(so)} so far → ${fmtUSD(cost.p50)} expected · worst ${fmtUSD(cost.worst)}`;
  return so != null ? fmtUSD(so) : "–";
}
