// Snapshots (#/snapshots and #/snapshots/<run>): the dashboard images taken at progress points of an analysis run or a pod
// session (work/pipeline/snapshots/<run>/, pipeline/snapshots.py). Newest first; click one to see it full size.
import { h, replace, fmtTime, fmtAgo, fmtDur, fmtUSD } from "../util.js";
import { errorText } from "../components/dialog.js";
import { thumb } from "./analysis.js";
import * as api from "../api.js";

export default {
  title: (r) => (r.run ? `Snapshots ${r.run}` : "Snapshots"),
  async mount(root, route) {
    const S = { run: route.run || null, timer: null, destroyed: false };
    const head = h("div", { class: "fm-head" });
    const body = h("div");
    replace(root, head, body);

    async function load() {
      try {
        if (!S.run) {
          const runs = await api.get("/api/snapshots");
          replace(head, h("div", { class: "fm-head-l" }, h("a", { href: "#/", class: "muted small" }, "← Farm"), h("h1", null, "Snapshots")));
          replace(body, runs.length ? h("div", { class: "am-thumbs" }, runs.map((r) => h("a", { class: "am-runlink card", href: `#/snapshots/${encodeURIComponent(r.run)}` },
            h("strong", { class: "mono" }, r.run), h("div", { class: "muted small" }, `${r.n} snapshot(s) · last ${fmtAgo(r.last.ts)} · ${r.last.label || r.last.event}`),
            r.last.png ? h("img", { src: `/api/snapshots/${encodeURIComponent(r.run)}/${encodeURIComponent(r.last.png)}`, alt: "", loading: "lazy" }) : null)))
            : h("p", { class: "muted" }, "No snapshots yet. Every analysis run and every pod session takes them at its progress points."));
          return;
        }
        const ix = await api.get(`/api/snapshots/${encodeURIComponent(S.run)}`);
        const shots = (ix.shots || []).slice().reverse();
        const isAnalysis = ix.kind === "analysis";
        replace(head, h("div", { class: "fm-head-l" }, h("a", { href: "#/snapshots", class: "muted small" }, "← all snapshots"),
          h("h1", null, `Snapshots of ${S.run}`)),
        h("div", { class: "fm-actions" }, isAnalysis ? h("a", { class: "btn", href: `#/analysis/${encodeURIComponent(S.run)}` }, "Open the analysis")
          : h("a", { class: "btn", href: `#/runpod/session/${encodeURIComponent(S.run)}` }, "Open the session")));
        replace(body, h("p", { class: "muted small" }, `${shots.length} snapshot(s) in work/pipeline/snapshots/${S.run}/ — static pages rendered to PNG by headless Edge at each progress point.`),
          shots.length ? h("div", { class: "am-thumbs am-thumbs-big" }, shots.map((s2) => thumb(S.run, s2))) : h("p", { class: "muted" }, "None yet."));
      } catch (e) {
        replace(body, h("div", { class: "fm-empty" }, errorText(e)));
      }
    }
    const schedule = () => { clearTimeout(S.timer); if (!S.destroyed) S.timer = setTimeout(async () => { await load(); schedule(); }, 15000); };
    await load();
    schedule();
    return { destroy() { S.destroyed = true; clearTimeout(S.timer); }, route(r) { S.run = r.run || null; load(); }, refresh() { load(); } };
  },
};
