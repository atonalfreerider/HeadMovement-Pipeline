// One analysis run (#/analysis/<run>): from "here are the videos" to the finished analysis (pipeline/analyze.py).
// KPIs (steps, elapsed, ETA with the finish time, cost so far / expected / p90, budget), the analysis map (one row per step,
// the defrag look), the pod sessions of the run with their uploads, phases, ETA and cost, the events, and the snapshots
// taken at its progress points. Polls GET /api/analyses/<run> every 3 s.
import { h, replace, fmtDur, fmtUSD, fmtTime, fmtAgo } from "../util.js";
import { pill } from "../components/status.js";
import { openDialog, errorText } from "../components/dialog.js";
import { stepBlocks, blocks, legend, stepLine, STATUS_TEXT, STATUS_KIND, sparkline, uploadText, uploadBar, sessionStatePill, costText, framesText, choiceText } from "../components/stepmap.js";
import * as api from "../api.js";

const GPU_SHORT = (g) => (g || "").replace(/^NVIDIA /, "").replace(/GeForce /, "").replace(/-SXM4/, " SXM").replace(/[- ]?80GB( HBM3)?/, "");

/** The alerts of a run (activity.analysis_alerts: stalls, slow GPUs, gone runners, overruns, failed steps), as a bar at
 *  the top of the page: they were only inside the session rows before. */
export function alertBar(alerts, { max = 6 } = {}) {
  const al = alerts || [];
  if (!al.length) return null;
  return h("div", { class: "fm-alerts" }, al.slice(0, max).map((a) => h("div", { class: ["alert", "alert-" + (a.level === "red" ? "red" : a.level === "amber" ? "amber" : "info")], role: a.level === "red" ? "alert" : null },
    h("span", { class: "alert-icon", "aria-hidden": "true" }, { red: "✕", amber: "!", info: "i" }[a.level] || "i"),
    h("span", { class: "alert-msg" }, a.message),
    a.link && /^#\//.test(a.link) ? h("a", { class: "btn small", href: a.link }, "Open") : null)),
  al.length > max ? h("div", { class: "alert alert-more muted small" }, `+ ${al.length - max} more`) : null);
}

/** The sessions of one step: every id in full (they were cut to 18 characters, 3 per step); more than 3 fold away. */
function stepSessions(sids, step, opened) {
  if (!sids || !sids.length) return null;
  const link = (sid) => h("a", { href: `#/runpod/session/${encodeURIComponent(sid)}`, class: "mono am-sid" }, sid);
  if (sids.length <= 3) return h("div", { class: "small am-sids" }, sids.map(link));
  // the page re-renders every 3 s: an opened list stays open (opened = the view's set of open steps)
  return h("details", { class: "small am-sids", open: opened && opened.has(step) ? true : null,
    ontoggle: (e) => { if (opened) { if (e.target.open) opened.add(step); else opened.delete(step); } } },
  h("summary", null, `${sids.length} pod sessions`), sids.map(link));
}

export default {
  title: (r) => `Analysis ${r.run || ""}`,
  async mount(root, route, app) {
    const S = { run: route.run, data: null, timer: null, destroyed: false, fetchedAt: Date.now(), openSteps: new Set() };
    const head = h("div", { class: "fm-head" });
    const sub = h("div", { class: "fm-subhead muted small" });
    const kpis = h("div", { class: "am-kpis" });
    const mapCard = h("section", { class: "card am-card" });
    const sessCard = h("section", { class: "card am-card" });
    const snapCard = h("section", { class: "card am-card" });
    const evCard = h("section", { class: "card am-card" });
    const covCard = h("section", { class: "card am-card", hidden: true });
    const alertsEl = h("div");
    replace(root, head, sub, alertsEl, kpis, mapCard, sessCard, covCard, snapCard, evCard);
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
      h("div", { class: "fm-actions" }, h("a", { class: "btn", href: `#/snapshots/${encodeURIComponent(d.run)}` }, `Snapshots (${d.snapshots_n != null ? d.snapshots_n : (d.snapshots || []).length})`)));
      replace(sub, `run ${d.run} · ${d.input} · ${f.n} video(s), ${Math.round(f.duration_s || 0)} s at ${f.fps} fps, ${Math.round(f.size_mb || 0)} MB · started ${fmtTime(d.created, true)}`);
      replace(alertsEl, alertBar(d.alerts));
      if (app && app.setTop) app.setTop(h("span", { class: "muted small" }, `${d.take} · ${d.status}`), (d.alerts || []).length
        ? pill(`${d.alerts.length} alert${d.alerts.length > 1 ? "s" : ""}`, d.alerts.some((a) => a.level === "red") ? "crit" : "warn") : null);
      const eta = lv.eta_s || {}, fin = lv.finish_at || {}, cost = lv.cost_usd || {};
      const kpi = (l, v, s2, cls) => h("div", { class: ["am-kpi", cls] }, h("div", { class: "am-kl" }, l), h("div", { class: "am-kv" }, v), h("div", { class: "am-ks muted" }, s2));
      const plan50 = `plan end to end ${fmtDur((plan.wall_min || {}).p50 * 60)} (p90 ${fmtDur((plan.wall_min || {}).p90 * 60)})`;
      replace(kpis,
        kpi("Steps", `${lv.done || 0}/${lv.total || 0}`, `${(lv.running || []).length} running · ${(lv.waiting_agent || []).length} need an agent${(lv.failed || []).length ? ` · ${lv.failed.length} failed` : ""}${(lv.reused || []).length ? ` · ${lv.reused.length} reused` : ""}`),
        // a finished run's elapsed time stops at its end (it kept counting: 1 h 52 for a run of 1 h 40)
        kpi("Elapsed", fmtDur(lv.ended ? lv.elapsed_s : now - d.created), lv.ended ? `ended ${fmtTime(lv.ended, true)} · ${plan50}` : plan50),
        kpi("Time left (ETA)", fin.p50 ? fmtDur(eta.p50) : "–", fin.p50 ? `p90 ${fmtDur(eta.p90)} · finish ~${fmtTime(fin.p50)} (p90 ${fmtTime(fin.p90)})` : "nothing left to run"),
        kpi("Pod cost", fmtUSD(cost.so_far), `expected ${fmtUSD(cost.p50)} · p90 ${fmtUSD(cost.p90)}${cost.limit != null ? ` · limit ${fmtUSD(cost.limit)}, ${fmtUSD(cost.left)} left` : ""}`),
        kpi("Critical path", (lv.critical || []).length ? `${lv.critical.length} steps` : "–", (lv.critical || []).join(" → ")));
      // the analysis map
      replace(mapCard, h("div", { class: "am-cardhead" }, h("h3", null, "Analysis map"), legend()),
        (d.steps || []).map((st) => h("div", { class: "am-row" },
          h("div", { class: "am-nm" }, h("strong", null, st.id), " ", pill(STATUS_TEXT[st.status] || st.status, STATUS_KIND[st.status] || "neutral"),
            st.reused ? [" ", pill("reused", "neutral", "an agent reused results that already existed: nothing was computed, no pod ran")] : null,
            h("div", { class: "muted small" }, st.label),
            h("div", { class: "muted small" }, `${st.reused ? "no pod (reused)" : st.where === "local" ? "this PC" : `${st.where} · ${st.lanes} pod${st.lanes > 1 ? "s" : ""}`}${st.executor === "agent" || st.executor === "farm" ? ` · ${st.executor}` : ""}${st.optional ? " · optional" : ""}`)),
          blocks(stepBlocks(st, now), `${st.id}: ${st.progress_frac ? `progress from ${st.progress_frac.basis}` : st.status === "running" ? "no progress figure: striped squares = time elapsed against the plan" : st.basis || ""}`),
          h("div", { class: "am-st" }, h("strong", null, stepLine(st, now)),
            st.status === "running" && st.progress_frac ? h("div", { class: "small" }, `${Math.round(100 * st.progress_frac.frac)} % · ${st.progress_frac.basis}`) : null,
            st.cost_usd ? h("div", { class: "muted small" }, `${fmtUSD(st.cost_usd, st.cost_usd < 0.01 ? 4 : 2)} of pods so far`)
              : !st.reused && st.cost_p50_usd && !["done", "skipped", "stopped"].includes(st.status) && !lv.ended
                ? h("div", { class: "muted small" }, `~${fmtUSD(st.cost_p50_usd)} of pods (p50)`) : null,
            st.progress && st.progress.total && st.progress.unit === "bytes" ? h("div", { class: "muted small" }, `${(st.progress.done / 1e6).toFixed(0)}/${(st.progress.total / 1e6).toFixed(0)} MB`) : null,
            st.message ? h("div", { class: ["small", "am-msg", st.status === "failed" ? "crit-text" : st.status === "waiting_agent" ? "warn-text" : "muted"], title: st.message }, st.message.length > 240 ? st.message.slice(0, 240) + " …" : st.message) : null,
            stepSessions(st.sessions, st.id, S.openSteps)))));
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
      podsInside(d.pods),
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
        if (app && app.noteMonitor && data.monitor) app.noteMonitor(data.monitor);
      } catch (e) {
        replace(kpis, h("div", { class: "fm-empty" }, h("strong", null, "Cannot read this analysis: "), errorText(e)));
      }
    }
    const schedule = () => { clearTimeout(S.timer); if (!S.destroyed) S.timer = setTimeout(async () => { await poll(); schedule(); }, document.visibilityState === "visible" ? 3000 : 15000); };
    // a tab that was hidden (15 s polls) is brought up to date the moment it is shown again
    const onVis = () => { if (document.visibilityState === "visible" && !S.destroyed) { poll(); schedule(); } };
    document.addEventListener("visibilitychange", onVis);
    await poll();
    schedule();
    return { destroy() { S.destroyed = true; clearTimeout(S.timer); document.removeEventListener("visibilitychange", onVis); }, route(r) { if (r.run !== S.run) { S.run = r.run; poll(); } }, refresh() {} };
  },
};

const wtext = (w) => (typeof w === "string" ? w : (w && (w.msg || w.message)) || "");

/** Pods a session created and deleted itself before its final pod (broken host, failed CUDA check, unreachable ssh):
 * part of the run's cost, in no session row's cost (the trial's $0.03 between the rows and the Pod cost KPI). */
function podsInside(pods) {
  const xs = (pods || []).filter((p) => p.how === "deleted inside its session");
  if (!xs.length) return null;
  const sum = xs.reduce((a, p) => a + (p.cost || 0), 0);
  return h("div", { class: "muted small am-wrap", title: "counted in the run's Pod cost, not in the session rows above" },
    `+ ${xs.length} pod${xs.length > 1 ? "s" : ""} deleted inside ${xs.length > 1 ? "their sessions" : "its session"} (broken host / CUDA check), ${fmtUSD(sum)}: `,
    xs.map((p) => `${p.sid || "?"} ${GPU_SHORT(p.gpu)} ${fmtUSD(p.cost)}`).join(" · "));
}

export function sessionRow(c) {
  const up = c.upload;
  const tag = c.analysis || {};
  const live = ["running", "preparing", "orphaned"].includes(c.state);
  const frames = (c.frames || []).filter((f) => live ? f.state === "running" || f.slow : f.slow).slice(0, 3);
  const warns = [...(c.warnings || []).map(wtext), ...(c.state_warnings || []).map(wtext)].filter(Boolean);
  return h("div", { class: ["am-sess", "am-sess-" + c.state] },
    h("div", { class: "am-sess-top" }, h("a", { class: "mono", href: `#/runpod/session/${encodeURIComponent(c.sid)}` }, c.sid),
      sessionStatePill(c), tag.step ? pill(tag.step, "neutral", "the analysis step this session belongs to") : null,
      h("span", { class: "muted small" }, [(c.jobs || []).join(", "), GPU_SHORT(c.gpu), c.price_hr ? `${fmtUSD(c.price_hr)}/h` : null].filter(Boolean).join(" · ")),
      h("span", { class: "spacer" }), h("span", { class: "small" }, costText(c))),
    c.phase ? h("div", { class: "small am-wrap" }, h("strong", null, c.phase), c.phase_detail ? h("span", { class: "muted" }, " · " + c.phase_detail) : null,
      c.eta && c.eta.p50 != null ? h("span", null, ` · ETA ${fmtDur(c.eta.p50)}${c.eta.p90 != null ? ` (p90 ${fmtDur(c.eta.p90)})` : ""}`) : null,
      c.progress ? h("span", { class: "muted" }, ` · ${Math.round(100 * c.progress.frac)} % (${c.progress.detail})`) : null) : null,
    frames.map((f) => h("div", { class: ["small", "am-wrap", f.slow ? "warn-text" : ""], title: "frame counter of the pod command (progress/frames.json)" },
      f.slow ? "slow: " : "frames: ", framesText(f))),
    // a session that never got a pod says why (it showed "44.2 MB to upload" and hid "every option exceeds the session max")
    c.no_pod ? h("div", { class: "crit-text small am-wrap" }, `no pod: ${c.error || "no GPU pod could be started"}`) : null,
    c.no_pod && c.choice ? h("div", { class: "muted small am-wrap" }, "tried: ", choiceText(c.choice)) : null,
    up && !c.no_pod ? h("div", { class: "am-up small" }, h("span", null, "upload: ", uploadText(up)), uploadBar(up), live ? sparkline(up.history) : null) : null,
    warns.slice(0, 3).map((w) => h("div", { class: "warn-text small am-wrap" }, w)),
    !c.no_pod && c.jobs_bad && Object.keys(c.jobs_bad).length ? h("div", { class: "crit-text small am-wrap" }, Object.entries(c.jobs_bad).map(([k, v]) => `${k}: ${v}`).join("; ").slice(0, 300)) : null,
    !c.no_pod && c.error && !(c.jobs_bad && Object.keys(c.jobs_bad).length) ? h("div", { class: "crit-text small am-wrap" }, c.error.slice(0, 300)) : null);
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
