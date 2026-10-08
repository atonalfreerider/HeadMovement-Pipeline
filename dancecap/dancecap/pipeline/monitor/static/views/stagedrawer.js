// Stage inspector drawer (#/take/<take>/stage/<id>, PIPELINE_MONITOR.md 6.6): status and reason, where it runs,
// what it does, dependencies, outputs (fresh / stale because), gates, estimate, camera-type behaviour, live progress,
// last runs with a log tail, review images and the actions.
import { h, replace, fmtTime, fmtDur, fmtEta, fmtCostBand, fmtAgo, fingerprint } from "../util.js";
import { getx, get, post, seg } from "../api.js";
import { statusBadge, statusGlyph, whereBadge, WHERE, pill } from "../components/status.js";
import { progressBar } from "../components/progress.js";
import { toast, errorText, confirmDialog } from "../components/dialog.js";
import { CLASS_LABEL } from "../components/cameraType.js";

export function openStageDrawer(app, take, stageId, onClose) {
  const back = h("div", { class: "drawer-back", onclick: () => onClose() });
  const head = h("div", { class: "drawer-head" });
  const body = h("div", { class: "drawer-body" }, h("div", { class: "loading" }, "loading…"));
  const acts = h("div", { class: "drawer-actions" });
  const box = h("aside", { class: "drawer", role: "dialog", "aria-label": `Stage ${stageId}` }, head, body, acts);
  const top = (document.querySelector(".topbar") || { getBoundingClientRect: () => ({ bottom: 46 }) }).getBoundingClientRect().bottom;
  box.style.top = back.style.top = `${Math.round(top)}px`;
  document.body.append(back, box);
  let alive = true;
  let fp = null;
  let review = null;
  const path = `/api/takes/${seg(take)}/stages/${seg(stageId)}`;

  async function load() {
    try {
      const { data } = await getx(path);
      if (!alive) return;
      const f = fingerprint(data);
      if (f === fp) return;
      fp = f;
      if (review === null) {
        try { review = await get(`/api/takes/${seg(take)}/review`); } catch { review = []; }
      }
      render(data);
    } catch (e) {
      if (!alive) return;
      replace(head, h("h2", null, stageId), h("button", { class: "icon-btn", onclick: () => onClose(), "aria-label": "Close" }, "✕"));
      replace(body, h("p", { class: "crit-text" }, errorText(e)));
    }
  }

  function render(d) {
    const def = d.def || {};
    replace(head,
      h("div", { style: { flex: 1, minWidth: 0 } },
        h("div", { class: "row" }, whereBadge(d.runs_on), h("h2", null, d.id), statusBadge(d.status)),
        h("div", { class: "muted", style: { marginTop: "3px" } }, d.name || def.name || ""),
        d.status_reason ? h("div", { class: "small", style: { marginTop: "4px" } }, d.status_reason) : null),
      h("button", { class: "icon-btn", title: "Close (Esc)", "aria-label": "Close", onclick: () => onClose() }, "✕"));

    const runs = d.runs || [];
    const active = runs.find((r) => ["running", "starting", "stopping"].includes(r.state)); // waiting_human is final
    const ct = d.camera_types;
    const behaviour = d.camera_type_behaviour || def.camera_types;
    const used = ct ? Object.entries(ct).filter(([, v]) => v && v.length) : [];
    const imgs = (d.review || []).length ? d.review : (review || []).filter((r) => r.kind === "image" && r.name.toLowerCase().includes(d.id.split("_")[0])).slice(0, 8).map((r) => r.name);
    replace(body,
      section("Where it runs",
        h("div", { class: "kv" },
          h("span", null, "host"), h("span", null, (WHERE[d.runs_on] || {}).label || d.runs_on,
            d.session ? [" · session ", h("span", { class: "chip session" }, d.session)] : null),
          h("span", null, "phase"), h("span", null, def.phase || String(d.column)),
          def.parallel ? h("span", null, "parallel") : null, def.parallel ? h("span", null, String(def.parallel)) : null,
          d.optional ? h("span", null, "optional") : null, d.optional ? h("span", null, "yes (only when included in a plan)") : null,
          h("span", null, "built"), h("span", null, d.implemented === false ? h("span", { class: "muted" }, "not yet (planned in PIPELINE.md)") : "yes"))),
      d.status === "running" || active ? section("Live", progressBar(d.progress, { eta: d.eta }),
        (d.progress_steps || []).length ? h("div", { class: "small", style: { marginTop: "6px" } },
          d.progress_steps.slice(-6).map((p) => h("div", { class: "row" }, h("span", { class: "chip" }, p.kind || p.step || ""),
            h("span", null, p.message || (p.total ? `${p.done}/${p.total} ${p.unit || ""}` : "")), h("span", { class: "muted" }, fmtTime(p.ts || p.t))))) : null) : null,
      section("What it does",
        def.notes ? h("p", null, def.notes) : null,
        (def.cli || []).length ? h("div", null, (Array.isArray(def.cli) ? def.cli : [def.cli]).map((c) => h("div", { class: "mono small", style: { overflowWrap: "anywhere" } }, c))) : null,
        (def.helps || []).length ? h("div", { class: "small", style: { marginTop: "6px" } }, h("span", { class: "muted" }, "Why it comes first: "),
          def.helps.map((x) => h("div", null, "→ ", h("strong", null, x.stage), ": ", x.why))) : null),
      section("Depends on",
        (d.deps_detail || []).length ? h("div", { class: "dep-list" }, d.deps_detail.map((x) =>
          h("a", { class: ["dep", x.soft ? "soft" : ""], href: `#/take/${take}/stage/${x.id}`, title: x.soft ? "soft dependency" : "" },
            statusGlyph(x.status), x.id))) : h("span", { class: "muted" }, "nothing")),
      section("Outputs",
        (d.outputs_detail || []).length ? h("table", { class: "tbl compact" }, h("tbody", null, d.outputs_detail.map((o) => h("tr", null,
          h("td", null, o.exists ? h("span", { class: "good-text" }, "✓") : h("span", { class: "muted" }, "–")),
          h("td", { class: "mono small", style: { overflowWrap: "anywhere" } }, o.path, o.count > 1 ? h("span", { class: "muted" }, ` (${o.count})`) : null),
          h("td", { class: "small muted num" }, o.mtime ? fmtAgo(o.mtime) : "missing"))))) : h("span", { class: "muted" }, "decided by a rule (no output files)"),
        d.outputs && (d.outputs.stale_because || []).length ? h("p", { class: "warn-text small" }, "stale because: ", d.outputs.stale_because.join("; ")) : null,
        d.outputs && (d.outputs.upstream_stale || []).length ? h("p", { class: "warn-text small" }, "upstream stale: ", d.outputs.upstream_stale.join(", ")) : null,
        (() => {
          const shown = new Set((d.outputs && d.outputs.upstream_stale || []).map((u) => `upstream stale: ${u}`));
          const ws = (d.warnings || []).filter((w) => !shown.has(w));
          return ws.length ? h("ul", { class: "small warn-text" }, ws.map((w) => h("li", null, w))) : null;
        })()),
      (d.gates || []).length ? section("Gates", d.gates.map((g) => h("div", { class: "row small" },
        g.open ? pill("open", "warn") : pill("closed", "good"), h("strong", null, g.id), h("span", { class: "muted" }, g.message || "")))) : null,
      section("Estimate",
        h("div", { class: "kv" },
          h("span", null, "time"), h("span", null, fmtEta(d.estimate)),
          h("span", null, "pod cost"), h("span", null, fmtCostBand(d.cost)),
          h("span", null, "confidence"), h("span", null, (d.estimate || {}).confidence || "–"),
          h("span", null, "basis"), h("span", { class: "small" }, (d.estimate || {}).basis || "–")),
        d.estimate && d.estimate.branches ? h("div", { class: "small", style: { marginTop: "6px" } }, "Branches: ",
          d.estimate.branches.map((b) => h("div", null, `• ${b.name}: ${fmtDur(b.eta_p50_s)}`))) : null),
      behaviour ? section("Camera types in this stage",
        used.length ? h("div", { class: "small", style: { marginBottom: "6px" } }, used.map(([k, v]) => h("div", null,
          h("strong", null, CLASS_LABEL[k] || k), ": cams ", v.join(", ")))) : null,
        h("div", { class: "behaviour" }, Object.entries(behaviour).map(([k, v]) => h("div", { class: ["b-row", used.some(([u]) => u === k) ? "used" : ""] },
          h("strong", null, CLASS_LABEL[k] || k), h("div", { class: "small" }, v)))),
        h("a", { class: "btn small", href: `#/take/${take}/videos`, style: { marginTop: "6px" } }, "Set camera types")) : null,
      runs.length ? section("Last runs", h("table", { class: "tbl compact" }, h("tbody", null, runs.map((r) => h("tr", null,
        h("td", null, h("a", { href: `#/runs/${r.id}` }, r.id)), h("td", null, pill(r.state, r.state === "done" ? "good" : r.state === "failed" ? "crit" : "info")),
        h("td", { class: "small muted" }, fmtTime(r.started || r.created)), h("td", { class: "small num" }, r.ended && r.started ? fmtDur(r.ended - r.started) : ""),
        h("td", { class: "small" }, r.rc !== null && r.rc !== undefined ? `rc ${r.rc}` : ""))))),
        (d.log_tail || []).length ? h("div", { class: "log-body", style: { height: "180px", marginTop: "8px" } }, d.log_tail.slice(-60).map((l) => h("div", null, l))) : null) : null,
      imgs.length ? section("Review images", h("div", { class: "review-imgs" }, imgs.map((name) => {
        const n = String(name).replace(/^review\//, "");
        const url = `/api/takes/${seg(take)}/review/${n.split("/").map(encodeURIComponent).join("/")}`;
        return h("a", { href: url, target: "_blank", rel: "noopener" }, h("img", { src: url, alt: n, loading: "lazy" }), h("div", { class: "cap ellipsis" }, n));
      }))) : null);

    const A = async () => app.actions();
    const runnable = d.runs_on !== "human" && d.implemented !== false;
    // a review gate the orchestrator opened after running this stage (fit_qa: ok | reject; geoveto: apply | keep |
    // reject): POST /gates/<stage> {decision}; reject marks the stage stale (re-run it or its inputs)
    function reviewGateButtons(st) {
      const g = (st.gates || []).find((x) => x.id === "review" && x.open);
      if (!g) return null;
      const label = { ok: "Looks good", apply: "Apply the veto", keep: "Keep as measured", reject: "Reject" };
      return (g.decisions || ["ok", "reject"]).map((dec) => h("button", { class: ["btn", dec === "reject" ? "danger" : "primary"],
        title: g.message || "", onclick: async () => {
          if (!await confirmDialog(`${label[dec] || dec}: ${st.id}?`, g.message || "", { okLabel: label[dec] || dec, danger: dec === "reject" })) return;
          try {
            const r = await post(`/api/takes/${seg(take)}/gates/${seg(st.id)}`, { decision: dec });
            toast(r.run ? `Review confirmed (${dec}); the veto is applied in run ${r.run.id}.` : `Review confirmed (${dec}).`, "good");
            fp = null; load(); app.refreshNow();
          } catch (e) { toast(errorText(e), "crit"); }
        } }, label[dec] || dec));
    }
    replace(acts,
      runnable ? h("button", { class: "btn primary", onclick: async () => (await A()).planDialog(app, take, { target: d.id, scope: "only" }) }, "Run only this…") : null,
      runnable ? h("button", { class: "btn", onclick: async () => (await A()).planDialog(app, take, { target: d.id, scope: "upto" }) }, "Run up to here…") : null,
      d.status === "stale" && !(d.outputs && (d.outputs.stale_because || []).some((x) => /camera/i.test(x))) ? h("button", { class: "btn",
        title: "Keep these outputs: an upstream rewrite that does not matter", onclick: async () => {
          if (!await confirmDialog(`Keep ${d.id}'s outputs?`, "The stage counts as fresh again until one of its inputs changes.", { okLabel: "Keep outputs" })) return;
          try { await post(`/api/takes/${seg(take)}/stages/${seg(d.id)}/accept-inputs`, {}); toast("Marked fresh.", "good"); fp = null; load(); app.refreshNow(); }
          catch (e) { toast(errorText(e), "crit"); }
        } }, "Keep outputs") : null,
      d.id === "camera_types" ? h("a", { class: "btn primary", href: `#/take/${take}/videos` }, "Review camera types") : null,
      d.id === "narrate" ? h("button", { class: "btn primary", onclick: async () => (await A()).narrate(app, take) }, "Narrate moves") : null,
      d.id === "labels_review" && d.status !== "done" ? h("button", { class: "btn primary", onclick: async () => {
        try { await post(`/api/takes/${seg(take)}/gates/labels_review`, {}); toast("Labels review confirmed.", "good"); fp = null; load(); app.refreshNow(); }
        catch (e) { toast(errorText(e), "crit"); }
      } }, "I looked at the labels sheet") : null,
      d.id === "demo_export" ? h("button", { class: "btn", onclick: async () => (await A()).exportDialog(app, take) }, "Export…") : null,
      reviewGateButtons(d),
      active ? h("button", { class: "btn danger", onclick: async () => (await A()).stopRun(app, active) }, "Stop") : null,
      h("span", { class: "spacer" }),
      h("button", { class: "btn ghost", onclick: async () => (await A()).reveal(take, "work") }, "Reveal outputs"));
  }

  function section(title, ...kids) {
    return h("section", null, h("h4", null, title), ...kids);
  }

  load();
  return {
    refresh: load,
    destroy() { alive = false; back.remove(); box.remove(); },
  };
}
