// Runs (#/runs, #/runs/<id>): every process the monitor or `pipeline run` started (+ external ones), with state,
// progress, stop modes and the live log.
import { h, replace, fmtTime, fmtDur, fingerprint } from "../util.js";
import { getx, get, post, seg, listOf } from "../api.js";
import { pill } from "../components/status.js";
import { progressBar } from "../components/progress.js";
import { toast, errorText, confirmDialog } from "../components/dialog.js";
import { logView } from "../components/logview.js";
import { runsTable } from "./take.js";

const STOP_TEXT = {
  after_step: ["Stop after this step", "The run finishes its current stage, then stops."],
  now: ["Stop now", "Interrupts the run; its cleanup (finally blocks) still runs. Outputs are written atomically, so a half stage leaves no broken files."],
};

export default {
  title: (r) => (r.id ? r.id : "Runs"),
  async mount(root, route, app) {
    let id = route.id;
    const listEl = h("div");
    const detailEl = h("div", { style: { marginTop: "14px" } });
    replace(root, h("div", { class: "page-head" }, h("h1", null, "Runs"),
      h("span", { class: "muted" }, "processes started by the monitor or the pipeline CLI; other workflows' runs appear as external (read-only)")), listEl, detailEl);
    let lv = null, fpL = null, fpD = null;

    async function loadList() {
      let runs;
      try { runs = listOf((await getx("/api/runs?limit=100")).data, "runs"); }
      catch (e) { replace(listEl, h("p", { class: "crit-text" }, errorText(e))); return; }
      const f = fingerprint({ runs, id });
      if (f === fpL) return;
      fpL = f;
      replace(listEl, runs.length ? runsTable(runs) : h("div", { class: "empty" }, "No runs yet. Start one from a take (Plan & run, Narrate moves, Export)."));
      for (const tr of listEl.querySelectorAll("tbody tr")) tr.classList.toggle("sel", tr.firstChild && tr.firstChild.textContent === id);
    }

    async function loadDetail() {
      if (!id) { replace(detailEl); if (lv) { lv.destroy(); lv = null; } return; }
      let r;
      try { r = await get(`/api/runs/${seg(id)}`); }
      catch (e) { replace(detailEl, h("div", { class: "card crit-text" }, errorText(e))); return; }
      const f = fingerprint(r);
      if (f === fpD) return;
      const first = fpD === null;
      fpD = f;
      const active = ["running", "starting", "stopping"].includes(r.state); // waiting_human: ended at a gate (final)
      const info = h("div", { class: "card" },
        h("div", { class: "card-head" }, h("h2", { class: "mono" }, r.id),
          pill(r.state, { running: "info", done: "good", failed: "crit", lost: "crit", stopped: "warn", waiting_human: "warn" }[r.state] || "neutral"),
          r.url ? h("button", { class: "btn small primary", onclick: async () => {
            try { await post("/api/shell/open", { tool_run: r.id }); } catch (e) { toast(errorText(e), "warn"); }
          } }, "Open tool window") : null,
          (r.stop_modes || []).map((m) => h("button", { class: "btn small danger", onclick: async () => {
            const [label, text] = STOP_TEXT[m] || [m, ""];
            if (!await confirmDialog(`${label}: ${r.id}?`, text, { okLabel: label, danger: true })) return;
            try { await post(`/api/runs/${seg(r.id)}/stop`, { mode: m }); toast("Stop requested.", "good"); fpD = null; loadDetail(); }
            catch (e) { toast(errorText(e), "crit"); }
          } }, (STOP_TEXT[m] || [m])[0])),
          r.take ? h("button", { class: "btn small ghost", onclick: async () => (await app.actions()).reveal(r.take, "run_log", r.id) }, "Reveal log") : null),
        active ? progressBar(r.progress, { eta: r.progress && r.progress.eta,
          label: r.current ? [r.current.stage || "", r.current.step ? h("span", { class: "muted" }, " · " + r.current.step) : null] : null }) : null,
        h("div", { class: "kv", style: { marginTop: "10px" } },
          h("span", null, "kind"), h("span", null, r.kind),
          h("span", null, "take"), h("span", null, r.take ? h("a", { href: `#/take/${r.take}` }, r.take) : "–"),
          h("span", null, "stages"), h("span", null, (r.stages || []).join(", ") || "–"),
          h("span", null, "command"), h("span", { class: "mono small", style: { overflowWrap: "anywhere" } }, (r.argv_display || []).join(" ")),
          h("span", null, "origin"), h("span", null, r.origin || "–"),
          h("span", null, "pid"), h("span", null, r.pid ? String(r.pid) : "–"),
          h("span", null, "started"), h("span", null, fmtTime(r.started || r.created, true)),
          h("span", null, "duration"), h("span", null, r.started ? fmtDur((r.ended || Date.now() / 1000) - r.started) : "–"),
          h("span", null, "exit code"), h("span", null, r.rc !== null && r.rc !== undefined ? String(r.rc) : "–"),
          r.current && r.current.session_sid ? h("span", null, "session") : null,
          r.current && r.current.session_sid ? h("a", { href: `#/runpod/session/${r.current.session_sid}`, class: "mono" }, r.current.session_sid) : null,
          r.plan_hash ? h("span", null, "plan") : null, r.plan_hash ? h("span", { class: "mono small" }, r.plan_hash) : null));
      if (first || !lv) {
        if (lv) lv.destroy();
        lv = logView({ url: `/api/runs/${seg(r.id)}/log`, title: "output.log", height: "420px" });
        replace(detailEl, info, h("div", { style: { marginTop: "12px" } }, lv.el));
      } else detailEl.replaceChild(info, detailEl.firstChild);
    }

    await loadList();
    await loadDetail();
    return {
      async refresh() { await loadList(); await loadDetail(); },
      route(r) { id = r.id; fpL = null; fpD = null; if (lv) { lv.destroy(); lv = null; } loadList(); loadDetail(); },
      destroy() { if (lv) lv.destroy(); },
    };
  },
};
