// Progress bars and the budget gauge (spent -> p50 -> p90 against cap - reserve and the cap).
import { h, fmtUSD, fmtEta, clamp } from "../util.js";

/**
 * progress: {done, total, unit, frac, step} (any may be missing). Running bars animate (static under
 * prefers-reduced-motion); a missing fraction gives an indeterminate bar.
 */
export function progressBar(progress, { running = true, eta = null, compact = false, label = null } = {}) {
  const p = progress || {};
  let frac = p.frac;
  if ((frac === undefined || frac === null) && p.total) frac = (p.done || 0) / p.total;
  const known = frac !== undefined && frac !== null && isFinite(frac);
  const bar = h("div", {
    class: ["pbar", running ? "running" : "", known ? "" : "indeterminate", compact ? "compact" : ""],
    role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100",
    "aria-valuenow": known ? String(Math.round(frac * 100)) : null,
  }, h("div", { class: "fill", style: { width: known ? `${clamp(frac, 0, 1) * 100}%` : "35%" } }));
  if (compact) return bar;
  const txt = label ?? [
    p.step ? h("span", { class: "muted" }, p.step + "  ") : null,
    p.total ? `${p.done ?? 0}/${p.total} ${p.unit || ""}` : p.msg ? p.msg : known ? `${Math.round(frac * 100)} %` : "working…",
    p.total && p.msg ? h("span", { class: "muted" }, "  " + p.msg) : null,
  ];
  return h("div", { class: "pwrap" }, bar,
    h("div", { class: "pmeta" }, h("span", null, txt), eta ? h("span", { class: "eta" }, "ETA " + fmtEta(eta)) : null));
}

/**
 * Budget gauge. b: {spent, p50, p90, worst?, cap, reserve, level?}. All dollars.
 * Segments: spent (solid) | +p50 (medium) | +p90 (light, hatched); ticks at cap - reserve and cap.
 */
export function budgetGauge(b, { title = "Budget", compact = false } = {}) {
  const cap = b.cap ?? 30, reserve = b.reserve ?? 2;
  const spent = b.spent ?? 0, p50 = Math.max(spent, b.p50 ?? spent), p90 = Math.max(p50, b.p90 ?? p50);
  const worst = b.worst && b.worst > p90 ? b.worst : null;
  const max = Math.max(cap, worst || 0, p90) * 1.02;
  const x = (v) => `${clamp(v / max, 0, 1) * 100}%`;
  const w = (a, z) => `${clamp((z - a) / max, 0, 1) * 100}%`;
  const level = b.level || (p90 > cap ? "red" : p90 > cap - reserve ? "amber" : "green");
  const levelIcon = { green: "✓", amber: "!", red: "✕" }[level] || "";
  const bar = h("div", { class: ["gauge", compact ? "compact" : ""], role: "img",
    "aria-label": `spent ${fmtUSD(spent)}, projected ${fmtUSD(p50)} (p90 ${fmtUSD(p90)}), cap ${fmtUSD(cap)}` },
    h("div", { class: "g-seg g-spent", style: { left: "0", width: x(spent) }, title: `spent ${fmtUSD(spent)}` }),
    p50 > spent ? h("div", { class: "g-seg g-p50", style: { left: x(spent), width: w(spent, p50) },
      title: `+ expected (p50) ${fmtUSD(p50 - spent)}` }) : null,
    p90 > p50 ? h("div", { class: "g-seg g-p90", style: { left: x(p50), width: w(p50, p90) },
      title: `+ pessimistic (p90) ${fmtUSD(p90 - p50)}` }) : null,
    worst ? h("div", { class: "g-tick g-worst", style: { left: x(worst) }, title: `hard worst case (kill switches) ${fmtUSD(worst)}` }) : null,
    h("div", { class: "g-tick g-reserve", style: { left: x(cap - reserve) }, title: `cap - reserve ${fmtUSD(cap - reserve)}` }),
    h("div", { class: "g-tick g-cap", style: { left: x(cap) }, title: `cap ${fmtUSD(cap)}` }),
  );
  if (compact) return h("div", { class: "gauge-compact", title: `${title}: ${fmtUSD(spent)} of ${fmtUSD(cap)}` },
    h("span", { class: "gc-txt" }, `${fmtUSD(spent)} / ${fmtUSD(cap, 0)}`), bar);
  return h("div", { class: "gauge-wrap" },
    h("div", { class: "gauge-head" },
      h("span", { class: "gauge-title" }, title),
      h("span", { class: ["lvl", "lvl-" + level] }, h("span", { "aria-hidden": "true" }, levelIcon), " ",
        { green: "within budget", amber: "over cap - reserve", red: "over the cap" }[level] || level)),
    bar,
    h("div", { class: "gauge-legend" },
      h("span", null, h("i", { class: "sw g-spent" }), `spent ${fmtUSD(spent)}`),
      p50 > spent ? h("span", null, h("i", { class: "sw g-p50" }), `expected ${fmtUSD(p50)}`) : null,
      p90 > p50 ? h("span", null, h("i", { class: "sw g-p90" }), `p90 ${fmtUSD(p90)}`) : null,
      worst ? h("span", null, h("i", { class: "sw g-worst" }), `worst ${fmtUSD(worst)}`) : null,
      h("span", null, h("i", { class: "sw g-reserve" }), `cap - reserve ${fmtUSD(cap - reserve)}`),
      h("span", null, h("i", { class: "sw g-cap" }), `cap ${fmtUSD(cap)}`)));
}

/** Normalise the server's Budget object (PIPELINE_PROGRESS 8) for the gauge. */
export function budgetFromState(bud, plan) {
  if (!bud) return { spent: 0, cap: 30, reserve: 2 };
  const proj = bud.projected || {};
  const out = {
    spent: bud.spent ?? 0, cap: bud.cap ?? 30, reserve: bud.reserve ?? 2, level: bud.level,
    p50: proj.t50 ?? bud.spent, p90: proj.t90 ?? bud.spent, worst: proj.worst,
  };
  if (plan && plan.budget) {
    out.p50 = Math.max(out.p50, plan.budget.after_p50 ?? out.p50);
    out.p90 = Math.max(out.p90, plan.budget.after_p90 ?? out.p90);
    out.level = { ok: "green", over_reserve: "amber", over_cap: "red" }[plan.budget.gate] || out.level;
  }
  return out;
}
