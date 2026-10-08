// Session view (#/runpod/session/<sid>): the PIPELINE_PROGRESS snapshot of one remote_jobs session: current step
// with progress + ETA, phase Gantt (actual vs remaining), cost so far / p50 / p90 / worst, kill-switch margin,
// warnings, and the runner log.
import { h, replace, fmtTime, fmtDur, fmtUSD, fmtEta, fingerprint } from "../util.js";
import { getx, post, seg } from "../api.js";
import { connPill, pill } from "../components/status.js";
import { progressBar } from "../components/progress.js";
import { toast, errorText, typedConfirm } from "../components/dialog.js";
import { logView } from "../components/logview.js";
import { sessionStatePill } from "./runpod.js";

function gantt(snap) {
  const phases = (snap.phases || []).filter((p) => p.start || p.status === "running" || p.status === "pending");
  if (!phases.length) return h("div", { class: "empty" }, "No phases recorded.");
  const now = snap.now || Date.now() / 1000;
  const starts = phases.map((p) => p.start).filter(Boolean);
  const t0 = Math.min(...starts);
  let t1 = Math.max(...phases.map((p) => p.end || (p.status === "running" ? now + ((p.remaining && p.remaining.p50) || 0) : 0)), now > t0 && snap.state === "running" ? now : 0);
  let cursor = t1;
  for (const p of phases) if (!p.start && p.status === "pending") cursor += (p.remaining && p.remaining.p50) || 0;
  t1 = Math.max(t1, cursor, t0 + 1);
  const pct = (t) => `${((t - t0) / (t1 - t0)) * 100}%`;
  let pend = Math.max(now, ...phases.map((p) => p.end || 0));
  return h("div", { class: "gantt" }, phases.map((p) => {
    let bar;
    if (p.start) {
      const end = p.end || now;
      const cls = p.status === "failed" ? "failed" : p.status === "running" ? "running" : p.kind === "upload" ? "upload" : "";
      bar = [h("div", { class: ["g-bar", cls], style: { left: pct(p.start), width: `calc(${pct(end)} - ${pct(p.start)})` },
        title: `${p.name}: ${fmtDur(p.elapsed_s ?? end - p.start)}` }),
      p.status === "running" && p.remaining && p.remaining.p50 ? h("div", { class: "g-bar rest", style: { left: pct(end), width: `calc(${pct(end + p.remaining.p50)} - ${pct(end)})` },
        title: `remaining p50 ${fmtDur(p.remaining.p50)} (p90 ${fmtDur(p.remaining.p90)})` }) : null];
    } else {
      const d = (p.remaining && p.remaining.p50) || 0;
      bar = h("div", { class: "g-bar pending", style: { left: pct(pend), width: `calc(${pct(pend + d)} - ${pct(pend)})` }, title: `estimated ${fmtDur(d)}` });
      pend += d;
    }
    return h("div", { class: "g-row" },
      h("span", { class: ["ellipsis", p.optional ? "muted" : ""], title: p.name }, p.name),
      h("div", { class: "g-track" }, bar),
      h("span", { class: "num small" }, p.status === "running" ? `ETA ${fmtDur(p.remaining && p.remaining.p50)}` : p.elapsed_s !== null && p.elapsed_s !== undefined ? fmtDur(p.elapsed_s) : p.status));
  }), h("div", { class: "row small muted" }, h("span", null, fmtTime(t0)), h("span", { class: "spacer" }),
    h("span", null, "orange = upload (billed GPU time) · blue hatched = remaining estimate · dashed = not started"), h("span", { class: "spacer" }), h("span", null, fmtTime(t1))));
}

export function currentText(snap) {
  const c = snap && snap.current;
  if (!c) return null;
  return typeof c === "string" ? c : c.step || c.phase || null;
}

export default {
  title: (r) => r.sid,
  async mount(root, route, app) {
    const sid = route.sid;
    const top = h("div");
    const logHost = h("div", { class: "card" });
    replace(root, top, logHost);
    const lv = logView({ url: `/api/sessions/${seg(sid)}/log`, title: "runner.log", height: "380px" });
    logHost.appendChild(lv.el);
    let fp = null;
    async function load() {
      let snap;
      try { snap = (await getx(`/api/sessions/${seg(sid)}`)).data; }
      catch (e) { replace(top, h("div", { class: "card crit-text" }, e.status === 404 ? `No session ${sid}.` : errorText(e))); return; }
      const f = fingerprint(snap);
      if (f === fp) return;
      fp = f;
      const sm = snap.summary || {};
      const pod = snap.pod || {};
      const cost = snap.cost || {};
      const eta = snap.eta;
      replace(top,
        h("div", { class: "page-head" },
          h("a", { href: "#/runpod", class: "btn small ghost" }, "← Runpod"),
          h("h1", { class: "mono" }, sid), sessionStatePill(sm.state || snap.state),
          pill(snap.origin || sm.origin || "?", "neutral", "external = started outside the monitor (shown read-only)"),
          snap.take ? h("a", { href: `#/take/${snap.take}` }, snap.take) : null,
          h("span", { class: "muted small" }, `progress level ${snap.progress_level || snap.level || "L0"}`),
          eta && eta.basis ? h("span", { class: "muted small" }, `ETA basis: ${eta.basis}`) : null),
        h("div", { class: "metrics", style: { marginBottom: "12px" } },
          h("div", { class: "metric", style: { minWidth: "320px", flex: 1 } }, h("div", { class: "m-label" }, "Now"),
            snap.progress ? progressBar(snap.progress, { running: snap.state === "running", eta: eta && { p50_s: eta.p50, p90_s: eta.p90 } })
              : h("div", { class: "m-value" }, currentText(snap) || (snap.state === "running" ? "working…" : snap.state || "–"))),
          h("div", { class: "metric" }, h("div", { class: "m-label" }, "Remaining"), h("div", { class: "m-value" }, eta ? fmtEta({ p50_s: eta.p50, p90_s: eta.p90 }) : "–")),
          h("div", { class: "metric" }, h("div", { class: "m-label" }, "Cost"), h("div", { class: "m-value" }, fmtUSD(cost.so_far)),
            h("div", { class: "m-sub" }, snap.state === "running" ? `p50 ${fmtUSD(cost.p50)} · p90 ${fmtUSD(cost.p90)} · worst ${fmtUSD(cost.worst)}` : `${fmtUSD(cost.price_per_hr)}/h`)),
          h("div", { class: "metric" }, h("div", { class: "m-label" }, "Kill switch"),
            h("div", { class: "m-value" }, snap.kill_at ? fmtTime(snap.kill_at) : "–"),
            h("div", { class: ["m-sub", snap.kill_margin !== null && snap.kill_margin !== undefined && snap.kill_margin < 0 ? "crit-text" : ""] },
              snap.kill_margin !== null && snap.kill_margin !== undefined ? snap.kill_margin < 0 ? `${fmtDur(Math.abs(snap.kill_margin))} short at the p90 ETA: the kill switch could fire before the last download`
                : `${fmtDur(snap.kill_margin)} margin after the last download (p90)` : `max ${sm.max_minutes || "?"} min`))),
        h("div", { class: "grid-main-side" },
          h("div", { class: "card" }, h("h3", { class: "section-title" }, "Phases"), gantt(snap)),
          h("div", null,
            h("div", { class: "card" }, h("h3", { class: "section-title" }, "Pod"),
              h("div", { class: "kv" },
                h("span", null, "id"), h("span", { class: "mono small" }, pod.id || sm.pod_id || "–"),
                h("span", null, "GPU"), h("span", null, pod.gpu || sm.gpu || "–"),
                h("span", null, "price"), h("span", null, pod.price_per_hr ? `${fmtUSD(pod.price_per_hr)}/h` : "–"),
                h("span", null, "connection"), h("span", null, pod.connection ? connPill(pod.connection, true) : "–"),
                h("span", null, "started"), h("span", null, fmtTime(sm.started, true)),
                h("span", null, "ended"), h("span", null, sm.ended ? fmtTime(sm.ended, true) : "–"),
                h("span", null, "jobs"), h("span", null, (sm.jobs || []).map((j) => `${j} (${(sm.jobs_status || {})[j] || "?"})`).join(", ") || "–"),
                h("span", null, "runner"), h("span", null, snap.runner_alive === true ? "alive" : snap.runner_alive === false ? h("span", { class: "crit-text" }, "gone") : "–")),
              (snap.stop_modes || []).includes("delete_pod") ? h("button", { class: "btn small danger", style: { marginTop: "10px" }, onclick: async () => {
                const tail = String(pod.id || sm.pod_id || "").slice(-4);
                const typed = await typedConfirm(`Delete the pod of ${sid}?`, h("p", { class: "warn-text" }, "The runner's next step fails and it cleans up; results not yet downloaded are lost."), tail, { okLabel: "Delete pod" });
                if (!typed) return;
                try { await post(`/api/sessions/${seg(sid)}/stop`, { mode: "delete_pod", confirm: typed }); toast("Pod deleted.", "good"); fp = null; load(); }
                catch (e) { toast(errorText(e), "crit"); }
              } }, "Delete pod…") : null),
            (snap.warnings || []).length ? h("div", { class: "card" }, h("h3", { class: "section-title" }, "Warnings"),
              h("ul", { class: "small warn-text", style: { paddingLeft: "18px", margin: 0 } }, snap.warnings.map((w) => h("li", null, typeof w === "string" ? w : w.message || JSON.stringify(w))))) : null,
            (snap.steps || []).length ? h("div", { class: "card" }, h("h3", { class: "section-title" }, "Steps"),
              snap.steps.map((st) => h("div", { style: { marginBottom: "6px" } }, progressBar(st, { running: st.status === "running", eta: st.eta, label: `${st.step || st.name}: ${st.done ?? ""}/${st.total ?? ""} ${st.unit || ""}` })))) : null)));
    }
    await load();
    return { refresh: load, destroy() { lv.destroy(); } };
  },
};
