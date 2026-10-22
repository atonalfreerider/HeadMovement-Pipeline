// The analysis map: one row per step of an analysis run, one square per pod x 2 min of the step's plan (the defrag look of
// the Farm page, applied to the whole analysis). Same rule as snapshots.step_blocks (pipeline/snapshots.py), so the live
// page and the PNG snapshots agree. Also: the compact strip (one square per step) and the upload sparkline.
import { h, s, fmtDur, fmtUSD, clamp } from "../util.js";
import { pill } from "./status.js";

export const STATUS_TEXT = { waiting_agent: "needs an agent", pending: "pending", running: "running", done: "done",
  failed: "failed", blocked: "blocked", skipped: "skipped", stopped: "stopped (run ended)" };
export const STATUS_KIND = { done: "good", running: "info", waiting_agent: "warn", failed: "crit", blocked: "crit", pending: "neutral", skipped: "neutral", stopped: "neutral" };

/** The measured progress of a running step (activity.step_progress: byte / frame counters, live pod session phases), else
 *  null: no progress figure. */
export function stepFrac(st) {
  const pf = st.progress_frac || {};
  if (pf.frac != null) return clamp(pf.frac, 0, 1);
  const pr = st.progress || {};
  if (pr.total) return clamp((pr.done || 0) / pr.total, 0, 1);
  return null;
}

/** Same rule as snapshots.step_blocks: measured progress fills "run" squares; without a figure the elapsed time against
 *  the plan is drawn as striped "time" squares and the time past the plan as extra "over" squares (the 2026-10-21 review
 *  saw 11 of 12 squares "done so far" on a step where nothing had run). A reused step is one grey square: no pods. */
export function stepBlocks(st, now, unitMin = 2, cap = 90) {
  if (st.reused) return ["reused"];
  const lanes = st.where !== "local" ? (st.lanes || 1) : 1;
  const per = Math.max(1, Math.round((st.p50_min || 1) / unitMin));
  const n = Math.min(cap, lanes * per);
  const status = st.status || "pending";
  if (status === "done") return Array(n).fill("done");
  if (status === "running") {
    const frac = stepFrac(st);
    if (frac != null) {
      const k = Math.round(frac * n);
      return Array(k).fill("run").concat(Array(n - k).fill("left"));
    }
    if (!st.started) return Array(n).fill("left");
    const ratio = (now - st.started) / 60 / Math.max(0.1, st.p50_min || 1);
    const k = Math.min(n, Math.round(Math.max(0, ratio) * n));
    const over = ratio > 1 ? Math.max(0, Math.min(cap - n, Math.round((ratio - 1) * n))) : 0;
    return Array(k).fill("time").concat(Array(n - k).fill("left"), Array(over).fill("over"));
  }
  const k = { waiting_agent: "wait", failed: "fail", blocked: "fail", skipped: "skip", stopped: "stop" }[status] || "n";
  return Array(n).fill(k);
}

export function blocks(kinds, title) {
  return h("div", { class: "am-sq", title }, kinds.map((k) => h("i", { class: ["am-b", k] })));
}

export function legend() {
  const sw = (k, t, title) => h("span", { class: "am-lg", title }, h("i", { class: ["am-b", k] }), t);
  return h("span", { class: "am-legend" }, sw("done", "done"),
    sw("run", "running: measured progress", "filled squares = the step's own byte or frame counter, or its live pod session's phases"),
    sw("time", "running, no progress figure: time elapsed vs plan", "no counter and no live pod session: the squares only say how much of the plan's time has passed"),
    sw("over", "past the plan"), sw("wait", "needs an agent"), sw("reused", "reused (not computed, no pods)"),
    sw("n", "pending"), sw("fail", "failed"), sw("stop", "stopped (the run ended)"), sw("skip", "skipped"),
    h("span", { class: "muted" }, "one square = one pod × 2 min of the plan"));
}

export function stepLine(st, now) {
  const status = st.status || "pending";
  if (st.reused) return "reused, not computed";
  let dur;
  if (st.started && st.ended) dur = `took ${fmtDur(st.ended - st.started)}`;
  else if (st.started && status === "running") dur = `running ${fmtDur(now - st.started)} of ~${Math.round(st.p50_min)} min`;
  else if (status === "stopped") dur = "stopped when the run ended";
  else dur = `plan ${Math.round(st.p50_min)} min (p90 ${Math.round(st.p90_min)})`;
  // the time left the run's ETA counts for this step (activity.analysis_card writes it back from analyze.live), so the
  // step row and the run KPI agree; "measured" = its pod session's measured progress covers the whole step
  if (status === "running" && st.eta_s != null) dur += ` · ETA ${fmtDur(st.eta_s)}${st.eta_s_p90 != null && st.eta_s_p90 > st.eta_s ? ` (p90 ${fmtDur(st.eta_s_p90)})` : ""}${st.eta_basis === "measured" ? " · measured" : ""}`;
  return dur;
}

/** One square per step (the strip on the Farm page). */
export function strip(steps) {
  return h("span", { class: "am-strip" }, (steps || []).map((st) => h("i", {
    class: ["am-b", st.reused ? "reused" : { done: "done", running: "run", waiting_agent: "wait", failed: "fail", blocked: "fail", skipped: "skip", stopped: "stop" }[st.status] || "n"],
    title: `${st.id}: ${st.reused ? "reused, not computed" : STATUS_TEXT[st.status] || st.status}${st.where ? " · " + st.where : ""}` })));
}

/** One frame counter of a pod command (progress/frames.json): "trial_sam3d_body/infer 120/599 frames · 0.355 s per
 *  person-frame · 2 min left", with the slowdown against the reference GPU when there is one. */
export function framesText(f) {
  const parts = [`${f.key}: ${f.done ?? "?"}${f.total ? "/" + f.total : ""} ${f.unit || "frames"}${f.stage ? ` (${f.stage}${f.stages_total ? "/" + f.stages_total : ""})` : ""}`];
  if (f.rate_s != null) parts.push(`${Number(f.rate_s).toPrecision(3)} s per ${f.rate_unit || f.unit || "frame"}`);
  if (f.eta_s != null && f.state === "running") parts.push(`${fmtDur(f.eta_s)} left`);
  if (f.state && f.state !== "running") parts.push(f.state);
  if (f.slow_x != null && f.ref) parts.push(`${f.slow_x.toFixed(1)}× the ${Number(f.ref.rate_s).toPrecision(3)} s measured on ${(f.ref.gpu || "the reference GPU").replace(/^NVIDIA /, "")}`);
  return parts.join(" · ");
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
  if (!up) return null;
  if (up.no_pod) return "nothing uploaded: no pod started";
  if (!up.total) return null;
  const mb = (b) => (b / 1e6).toFixed(b >= 1e8 ? 0 : 1);
  if (up.done) return `${mb(up.total)} MB uploaded${up.mbps ? ` at ${up.mbps.toFixed(2)} MB/s` : ""}${up.streams > 1 ? ` (${up.streams} streams)` : ""}`;
  if (up.failed) return `upload failed after ${mb(up.sent || 0)} of ${mb(up.total)} MB`;
  // an ended session's counter stopped where it was: no bar, no "left" (2026-10-21: "65/146 MB, 31 s left" an hour later)
  if (up.stopped) return `stopped at ${mb(up.sent || 0)} of ${mb(up.total)} MB (the session ended)`;
  if (up.sent == null) return `${mb(up.total)} MB to upload (${up.source || "waiting"})`;
  return `${mb(up.sent)}/${mb(up.total)} MB${up.mbps ? ` at ${up.mbps.toFixed(2)} MB/s` : ""}${up.eta_s != null ? `, ${fmtDur(up.eta_s)} left` : ""}${up.streams > 1 ? ` · ${up.streams} streams` : ""}`;
}

export function uploadBar(up) {
  if (!up || !up.total || up.done || up.stopped || up.no_pod || up.sent == null) return null;
  const f = clamp(up.sent / up.total, 0, 1);
  return h("div", { class: "fm-bar running", role: "progressbar", "aria-valuenow": String(Math.round(f * 100)), "aria-valuemin": "0", "aria-valuemax": "100" },
    h("div", { class: "fm-bar-fill", style: { width: `${f * 100}%` } }));
}

export function sessionStatePill(c) {
  // "unknown" with no result yet = a session still bundling / starting its pod (a --parallel -pN, a quiet gzip)
  const st = c.preparing ? "preparing" : c.state || "?";
  const kind = { running: "info", preparing: "info", ok: "good", failed: "crit", orphaned: "crit", ended: "neutral" }[st] || "neutral";
  const title = c.no_pod ? c.error || "no GPU pod could be started"
    : c.jobs_bad && Object.keys(c.jobs_bad).length ? Object.entries(c.jobs_bad).map(([k, v]) => `${k}: ${v}`).join("; ")
      : st === "orphaned" ? "the runner process is gone while the pod is open: nothing downloads its outputs" : null;
  return pill(st === "orphaned" ? "runner gone" : c.no_pod ? "no pod" : st, c.no_pod ? "crit" : kind, title);
}

/** Why a session runs on its GPU, or why it got none (state pod.choice / gpu_choice): one line per option tried. */
export function choiceText(ch) {
  if (!ch || !(ch.tried || []).length) return null;
  const short = (g) => (g || "?").replace(/^NVIDIA /, "").replace(/GeForce /, "");
  return (ch.tried || []).slice(0, 8).map((t) => `${short(t.gpu)} ${(t.cloud || "").toLowerCase()} ${t.price != null ? fmtUSD(t.price) + "/h" : ""}: ${t.result || "?"}`).join(" · ")
    + ((ch.tried || []).length > 8 ? ` · +${ch.tried.length - 8} more` : "");
}

export function costText(c) {
  const cost = c.cost || {};
  const so = cost.so_far != null ? cost.so_far : c.cost_usd;
  if (cost.p50 != null) return `${fmtUSD(so)} so far → ${fmtUSD(cost.p50)} expected · worst ${fmtUSD(cost.worst)}`;
  return so != null ? fmtUSD(so) : "–";
}
