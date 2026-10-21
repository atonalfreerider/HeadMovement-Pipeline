// One analysis run (#/analysis/<run>): from "here are the videos" to the finished analysis (pipeline/analyze.py).
// KPIs (steps, elapsed, ETA with the finish time, cost so far / expected / p90, budget), the analysis map (one row per step,
// the defrag look), the pod sessions of the run with their uploads, phases, ETA and cost, the events, and the snapshots
// taken at its progress points. Polls GET /api/analyses/<run> every 3 s.
import { h, replace, fmtDur, fmtUSD, fmtTime, fmtAgo } from "../util.js";
import { pill } from "../components/status.js";
import { openDialog, errorText } from "../components/dialog.js";
import { stepBlocks, blocks, legend, stepLine, STATUS_TEXT, STATUS_KIND, sparkline, uploadText, uploadBar, sessionStatePill, costText } from "../components/stepmap.js";
import * as api from "../api.js";

const GPU_SHORT = (g) => (g || "").replace(/^NVIDIA /, "").replace(/GeForce /, "");

export default {
  title: (r) => `Analysis ${r.run || ""}`,
  async mount(root, route, app) {
    const S = { run: route.run, data: null, timer: null, destroyed: false, fetchedAt: Date.now() };
    const head = h("div", { class: "fm-head" });
    const sub = h("div", { class: "fm-subhead muted small" });
    const kpis = h("div", { class: "am-kpis" });
    const mapCard = h("section", { class: "card am-card" });
    const sessCard = h("section", { class: "card am-card" });
    const snapCard = h("section", { class: "card am-card" });
    const evCard = h("section", { class: "card am-card" });
    const covCard = h("section", { class: "card am-card", hidden: true });
    replace(root, head, sub, kpis, mapCard, sessCard, covCard, snapCard, evCard);
    const srvNow = () => (S.data ? S.data.now_ts + (Date.now() - S.fetchedAt) / 1000 : Date.now() / 1000);

    function render(d) {
      S.data = d;
      const now = srvNow();
      const lv = d.live || {}, f = d.facts || {}, plan = d.plan || {};
      replace(head, h("div", { class: "fm-head-l" },
        h("a", { href: "#/", class: "muted small" }, "← Farm"),
        h("h1", null, `${d.take} — ${d.route === "mono" ? "monocular" : "multi-view"} analysis`),
        pill(d.status || "?", { running: "info", waiting: "warn", done: "good", failed: "crit", planned: "neutral", cancelled: "neutral" }[d.status] || "neutral",
          d.status === "waiting" ? "the runnable steps are done; the waiting ones need an agent (take-specific code today)" : null),
        d.driver_alive ? pill("driver running", "info", "python -m dancecap.pipeline analysis resume runs the next steps") : null),
      h("div", { class: "fm-actions" }, h("a", { class: "btn", href: `#/snapshots/${encodeURIComponent(d.run)}` }, `Snapshots (${(d.snapshots || []).length})`)));
      replace(sub, `run ${d.run} · ${d.input} · ${f.n} video(s), ${Math.round(f.duration_s || 0)} s at ${f.fps} fps, ${Math.round(f.size_mb || 0)} MB · started ${fmtTime(d.created, true)}`);
      const eta = lv.eta_s || {}, fin = lv.finish_at || {}, cost = lv.cost_usd || {};
      const kpi = (l, v, s2, cls) => h("div", { class: ["am-kpi", cls] }, h("div", { class: "am-kl" }, l), h("div", { class: "am-kv" }, v), h("div", { class: "am-ks muted" }, s2));
      replace(kpis,
        kpi("Steps", `${lv.done || 0}/${lv.total || 0}`, `${(lv.running || []).length} running · ${(lv.waiting_agent || []).length} need an agent${(lv.failed || []).length ? ` · ${lv.failed.length} failed` : ""}`),
        kpi("Elapsed", fmtDur(now - d.created), `plan end to end ${fmtDur((plan.wall_min || {}).p50 * 60)} (p90 ${fmtDur((plan.wall_min || {}).p90 * 60)})`),
        kpi("Time left (ETA)", fin.p50 ? fmtDur(eta.p50) : "–", fin.p50 ? `p90 ${fmtDur(eta.p90)} · finish ~${fmtTime(fin.p50)} (p90 ${fmtTime(fin.p90)})` : "nothing left to run"),
        kpi("Pod cost", fmtUSD(cost.so_far), `expected ${fmtUSD(cost.p50)} · p90 ${fmtUSD(cost.p90)}`),
        kpi("Critical path", (lv.critical || []).length ? `${lv.critical.length} steps` : "–", (lv.critical || []).join(" → ")));
      // the analysis map
      replace(mapCard, h("div", { class: "am-cardhead" }, h("h3", null, "Analysis map"), legend()),
        (d.steps || []).map((st) => h("div", { class: "am-row" },
          h("div", { class: "am-nm" }, h("strong", null, st.id), " ", pill(STATUS_TEXT[st.status] || st.status, STATUS_KIND[st.status] || "neutral"),
            h("div", { class: "muted small" }, st.label),
            h("div", { class: "muted small" }, `${st.where === "local" ? "this PC" : `${st.where} · ${st.lanes} pod${st.lanes > 1 ? "s" : ""}`}${st.executor === "agent" || st.executor === "farm" ? ` · ${st.executor}` : ""}${st.optional ? " · optional" : ""}`)),
          blocks(stepBlocks(st, now), `${st.id}: ${st.basis || ""}`),
          h("div", { class: "am-st" }, h("strong", null, stepLine(st, now)),
            st.cost_p50_usd ? h("div", { class: "muted small" }, `~${fmtUSD(st.cost_p50_usd)} of pods (p50)`) : null,
            st.progress && st.progress.total ? h("div", { class: "muted small" }, `${(st.progress.done / 1e6).toFixed(0)}/${(st.progress.total / 1e6).toFixed(0)} MB`) : null,
            st.message ? h("div", { class: ["small", st.status === "failed" ? "crit-text" : st.status === "waiting_agent" ? "warn-text" : "muted"], title: st.message }, st.message.slice(0, 160)) : null,
            st.sessions && st.sessions.length ? h("div", { class: "small" }, st.sessions.slice(0, 3).map((sid) => h("a", { href: `#/runpod/session/${encodeURIComponent(sid)}`, class: "mono" }, sid.slice(-18) + " "))) : null))));
      renderSessions(d, now);
      renderCoverage(d);
      renderSnaps(d);
      replace(evCard, h("div", { class: "am-cardhead" }, h("h3", null, "Events")),
        (d.events || []).length ? h("table", { class: "fm-tbl" }, h("tbody", null, d.events.slice(0, 25).map((e) => h("tr", null,
          h("td", { class: "muted num" }, fmtTime(e.ts)), h("td", null, h("span", { class: "chip" }, e.kind)), h("td", null, e.step || ""), h("td", null, e.msg || "")))))
          : h("p", { class: "muted" }, "No events yet."));
    }

    function renderSessions(d, now) {
      const cards = d.sessions || [];
      replace(sessCard, h("div", { class: "am-cardhead" }, h("h3", null, "Pod sessions of this run"),
        h("span", { class: "muted small" }, "tagged DANCECAP_ANALYSIS=" + d.run)),
      cards.length ? cards.map((c) => sessionRow(c)) : h("p", { class: "muted" }, "No pod session belongs to this run yet. An agent driving a step runs its pod jobs with DANCECAP_ANALYSIS and DANCECAP_ANALYSIS_STEP set; they appear here with their upload, phases, ETA and cost."),
      d.agent_hint && ((d.live || {}).waiting_agent || []).length ? h("p", { class: "muted small" }, d.agent_hint) : null);
    }

    function renderCoverage(d) {
      const cov = d.coverage;
      covCard.hidden = !(cov && cov.cells);
      if (!cov || !cov.cells) return;
      replace(covCard, h("div", { class: "am-cardhead" }, h("h3", null, "Final result of the take, one square per second (the farm's defrag map)"),
        h("a", { href: `#/farm/${encodeURIComponent(d.take)}`, class: "small" }, "open the Farm view")),
      h("div", { class: "am-sq am-cov" }, Array.from(cov.cells).slice(0, 1500).map((k) => h("i", { class: ["am-b", { c: "done", p: "wait", n: "n" }[k]] }))),
      h("div", { class: "muted small" }, `computed ${cov.updated || "?"}`));
    }

    function renderSnaps(d) {
      const shots = (d.snapshots || []).slice(-8).reverse();
      replace(snapCard, h("div", { class: "am-cardhead" }, h("h3", null, "Snapshots at progress points"),
        h("a", { href: `#/snapshots/${encodeURIComponent(d.run)}`, class: "small" }, "all")),
      shots.length ? h("div", { class: "am-thumbs" }, shots.map((s2) => thumb(d.run, s2))) : h("p", { class: "muted" }, "None yet: the driver takes one at the start, at every step's end and every 15 min."));
    }

    async function poll() {
      try {
        const data = await api.get(`/api/analyses/${encodeURIComponent(S.run)}`);
        data.now_ts = Date.now() / 1000;
        S.fetchedAt = Date.now();
        render(data);
      } catch (e) {
        replace(kpis, h("div", { class: "fm-empty" }, h("strong", null, "Cannot read this analysis: "), errorText(e)));
      }
    }
    const schedule = () => { clearTimeout(S.timer); if (!S.destroyed) S.timer = setTimeout(async () => { await poll(); schedule(); }, document.visibilityState === "visible" ? 3000 : 15000); };
    await poll();
    schedule();
    return { destroy() { S.destroyed = true; clearTimeout(S.timer); }, route(r) { if (r.run !== S.run) { S.run = r.run; poll(); } }, refresh() {} };
  },
};

export function sessionRow(c) {
  const up = c.upload;
  const tag = c.analysis || {};
  return h("div", { class: ["am-sess", "am-sess-" + c.state] },
    h("div", { class: "am-sess-top" }, h("a", { class: "mono", href: `#/runpod/session/${encodeURIComponent(c.sid)}` }, c.sid),
      sessionStatePill(c), tag.step ? pill(tag.step, "neutral", "the analysis step this session belongs to") : null,
      h("span", { class: "muted small" }, [(c.jobs || []).join(", "), GPU_SHORT(c.gpu), c.price_hr ? `${fmtUSD(c.price_hr)}/h` : null].filter(Boolean).join(" · ")),
      h("span", { class: "spacer" }), h("span", { class: "small" }, costText(c))),
    c.phase ? h("div", { class: "small" }, h("strong", null, c.phase), c.phase_detail ? h("span", { class: "muted" }, " · " + c.phase_detail) : null,
      c.eta && c.eta.p50 != null ? h("span", null, ` · ETA ${fmtDur(c.eta.p50)}${c.eta.p90 != null ? ` (p90 ${fmtDur(c.eta.p90)})` : ""}`) : null) : null,
    up ? h("div", { class: "am-up small" }, h("span", null, "upload: ", uploadText(up)), uploadBar(up), sparkline(up.history)) : null,
    (c.warnings || []).slice(0, 2).map((w) => h("div", { class: "warn-text small" }, w)),
    c.jobs_bad && Object.keys(c.jobs_bad).length ? h("div", { class: "crit-text small" }, Object.entries(c.jobs_bad).map(([k, v]) => `${k}: ${v}`).join("; ").slice(0, 240)) : null);
}

export function thumb(run, s2) {
  const src = s2.png ? `/api/snapshots/${encodeURIComponent(run)}/${encodeURIComponent(s2.png)}` : null;
  return h("button", { class: "am-thumb", title: `${s2.label || s2.event} · ${fmtTime(s2.ts, true)}`, onclick: () => openShot(run, s2) },
    src ? h("img", { src, alt: s2.label || s2.event, loading: "lazy" }) : h("div", { class: "am-noimg" }, s2.error || "no image"),
    h("div", { class: "small" }, h("strong", null, `#${s2.i}`), ` ${fmtTime(s2.ts)} · ${s2.label || s2.event}`));
}

export function openShot(run, s2) {
  const src = s2.png ? `/api/snapshots/${encodeURIComponent(run)}/${encodeURIComponent(s2.png)}` : null;
  openDialog({ title: `Snapshot #${s2.i}: ${s2.label || s2.event}`, wide: true,
    body: h("div", { class: "am-shot" }, src ? h("img", { src, alt: s2.label || s2.event }) : h("p", null, s2.error || "no image"),
      h("p", { class: "muted small" }, `${fmtTime(s2.ts, true)} · ${s2.method} ${s2.ms} ms · work/pipeline/snapshots/${run}/${s2.png || s2.html}`)),
    actions: [{ label: "Open the page", onClick: () => { window.open(`/api/snapshots/${encodeURIComponent(run)}/${encodeURIComponent(s2.html)}`, "_blank"); return false; } }, { label: "Close", kind: "primary" }] });
}
