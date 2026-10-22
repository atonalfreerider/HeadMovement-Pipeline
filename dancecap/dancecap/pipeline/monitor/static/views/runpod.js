// Runpod panel (#/runpod, PIPELINE_MONITOR.md 6.8): API connection, account, budget gauge with projection and cap,
// pods (read-only; delete only for dancecap pods after a typed confirmation), sessions.
import { h, replace, fmtTime, fmtDur, fmtUSD, fmtAgo, fingerprint } from "../util.js";
import { getx, post, seg, listOf } from "../api.js";
import { connPill, pill } from "../components/status.js";
import { budgetGauge, budgetFromState, progressBar } from "../components/progress.js";
import { toast, errorText, typedConfirm } from "../components/dialog.js";

const FLAG_TEXT = {
  orphan: "orphan: the pod is alive but its runner is gone",
  killswitch_unarmed: "kill switch not armed",
  untracked: "not in the ledger: untracked spend",
  phantom_accrual: "phantom accrual (ledger row open, pod gone)",
};

export function sessionStatePill(st) {
  return pill(st || "?", { ok: "good", running: "info", failed: "crit", dry_run: "neutral", stopped: "warn", ended: "neutral", interrupted: "warn" }[st] || "neutral");
}

/** Snapshots of the sessions behind running pods (live step, progress, ETA, cost, kill-switch margin). */
export async function activeSnapshots(rp) {
  const sids = new Set([...(rp.sessions_active || []), ...(rp.pods || []).map((p) => p.session && p.session.sid).filter(Boolean)]);
  const out = {};
  await Promise.all(Array.from(sids).map(async (sid) => {
    try { out[sid] = (await getx(`/api/sessions/${seg(sid)}`)).data; } catch { /* the session dir may be gone */ }
  }));
  return out;
}

/** Compact live view of one session snapshot: current step, progress bar with ETA, cost so far / p50 / p90. */
export function sessionLive(sn, { compact = false } = {}) {
  if (!sn) return null;
  const eta = sn.eta ? { p50_s: sn.eta.p50, p90_s: sn.eta.p90 } : null;
  const step = sn.current && typeof sn.current === "object" ? sn.current.step : sn.current;
  const prog = sn.progress || (step ? { step } : null);
  const c = sn.cost || {};
  return h("div", { class: "session-live" },
    progressBar(prog, { running: sn.state === "running", eta }),
    compact ? null : h("div", { class: "row small" },
      h("span", null, `cost ${fmtUSD(c.so_far)} so far`),
      c.p50 !== undefined ? h("span", { class: "muted" }, `→ ${fmtUSD(c.p50)} expected · ${fmtUSD(c.p90)} p90 · ${fmtUSD(c.worst)} worst`) : null,
      sn.kill_margin !== null && sn.kill_margin !== undefined && sn.kill_margin < 0
        ? h("span", { class: "crit-text", title: "the kill switch would fire before the last download at the p90 ETA" }, `kill-switch margin ${Math.round(sn.kill_margin / 60)} min at p90`) : null),
    compact ? null : (sn.warnings || []).slice(0, 2).map((w) => h("div", { class: "small warn-text" }, "! " + (typeof w === "string" ? w : w.message || ""))));
}

export async function deletePod(app, pod) {
  const tail = String(pod.id).slice(-4);
  const typed = await typedConfirm(`Delete pod ${pod.id}?`,
    [h("p", null, `${pod.name || "pod"} · ${pod.gpu || ""} · ${fmtUSD(pod.cost_per_hr)}/h`),
      h("p", { class: "warn-text" }, "Deleting stops the pod and anything running on it; its runner then cleans up and closes the ledger row. Results not yet downloaded are lost.")],
    tail, { okLabel: "Delete pod" });
  if (!typed) return false;
  try { await post(`/api/runpod/pods/${seg(pod.id)}/delete`, { confirm: typed }); toast(`Pod ${pod.id} deleted.`, "good"); app.refreshNow(); return true; }
  catch (e) { toast(errorText(e), "crit"); return false; }
}

/** dancecap pods the poller flags orphan: alive, an open ledger row, no active session, no runner process. */
export function orphanPods(pods) {
  return (pods || []).filter((p) => p.managed && !p.session && (p.flags || []).includes("orphan"));
}

/**
 * Orphan cleanup. The server's own orphan list (GET /api/runpod/orphans, fresh) is what the confirmation shows, and
 * exactly those ids are sent; the server deletes an id only if it is STILL an orphan. A pod whose runner is working
 * is never in the list, so this can never take a live session's pod down. An older server without that endpoint
 * (it deleted every dancecap-* pod) is refused here.
 */
export async function cleanupOrphans(app, after) {
  let list;
  try {
    list = (await getx("/api/runpod/orphans")).data.orphans || [];
  } catch (e) {
    if (e.status === 404) {
      toast("This monitor server predates the safe orphan cleanup: restart it (Settings > Restart monitor) first.", "warn");
    } else toast(errorText(e), "crit");
    return false;
  }
  if (!list.length) { toast("No orphan pods right now (every dancecap pod has a session or a runner).", "info"); if (after) after(); return false; }
  const ids = list.map((o) => o.id);
  const typed = await typedConfirm(`Delete ${ids.length} orphan pod${ids.length > 1 ? "s" : ""}?`,
    [h("p", null, "Only these pods are deleted: each is alive and billing, but no session and no runner process owns it."),
      h("table", { class: "tbl compact" }, h("tbody", null, list.map((o) => h("tr", null,
        h("td", { class: "mono" }, o.id), h("td", null, o.name || ""), h("td", { class: "num" }, `up ${fmtDur(o.age_s)}`),
        h("td", { class: "num" }, `${fmtUSD(o.cost_per_hr)}/h`), h("td", { class: "muted small" }, `ledger: ${o.ledger || "?"}`))))),
      h("p", { class: "warn-text" }, "If a pod turns out to have a session or a runner by the time you confirm, the server refuses and nothing is deleted.")],
    "DELETE", { okLabel: `Delete ${ids.length} pod${ids.length > 1 ? "s" : ""}` });
  if (!typed) return false;
  try {
    const r = await post("/api/runpod/cleanup", { confirm: typed, pods: ids });
    toast(`Deleted: ${(r.deleted || []).join(", ") || "none"}`, "good");
  } catch (e) {
    toast(errorText(e), e.status === 409 ? "warn" : "crit");
  }
  if (after) await after();
  app.refreshNow();
  return true;
}

const SETUP = new Set(["bundle_scan", "bundle_build", "pod_start", "extract", "bootstrap", "bootstrap_try1", "wheels_back", "delete"]);
function phaseCat(ph) {
  if (ph === "upload") return "upload";
  if (SETUP.has(ph)) return "setup";
  if (/(^|\/)(pack|download|extract)$/.test(ph)) return "results";
  return "work";
}

/** Where the money of the open sessions went: upload (billed: the pod exists while it uploads) / setup / work / results. */
function phaseCosts(pc) {
  if (!pc || !pc.length) return null;
  const groups = Array.isArray(pc[0].phases) ? pc : [{ sid: null, phases: pc }];
  const cats = ["upload", "setup", "work", "results"];
  return h("div", { style: { marginTop: "12px" } },
    h("h3", { class: "section-title" }, "Where the money goes (open sessions)"),
    h("table", { class: "tbl compact" },
      h("thead", null, h("tr", null, h("th", null, "session"), cats.map((c) => h("th", { class: "right" }, c)), h("th", { class: "right" }, "total"))),
      h("tbody", null, groups.map((g) => {
        const t = { upload: 0, setup: 0, work: 0, results: 0 };
        for (const ph of g.phases || []) t[phaseCat(ph.phase || ph.name || "")] += ph.usd ?? ph.cost_usd ?? 0;
        const tot = cats.reduce((a, c) => a + t[c], 0);
        return h("tr", null, h("td", { class: "mono small" }, g.sid || "–"),
          cats.map((c) => h("td", { class: ["num", "right", c === "upload" && tot && t[c] / tot > 0.25 ? "warn-text" : ""] },
            fmtUSD(t[c]), tot ? h("span", { class: "muted small" }, ` ${Math.round((t[c] / tot) * 100)}%`) : null)),
          h("td", { class: "num right" }, h("strong", null, fmtUSD(tot))));
      }))),
    h("p", { class: "small muted" }, "Upload time is paid GPU time (the pod is created before the bundle uploads)."));
}

function podCard(app, p, snaps = {}) {
  const k = p.kill || {};
  const flags = p.flags || [];
  return h("div", { class: ["pod-card", flags.length ? "flagged" : ""] },
    h("div", { class: "pod-head" },
      h("strong", null, p.name || "(unnamed)"), h("span", { class: "mono small muted", title: p.id }, "…" + String(p.id).slice(-6)),
      p.managed ? pill("dancecap", "info") : pill("not managed by dancecap", "neutral"),
      h("span", null, p.gpu || (p.session && snaps[p.session.sid] && snaps[p.session.sid].pod && snaps[p.session.sid].pod.gpu) || "GPU ?"),
      h("span", { class: "muted" }, (p.cloud || "").toLowerCase()),
      h("span", null, `${fmtUSD(p.cost_per_hr)}/h`),
      h("span", { class: "spacer" }),
      connPill(p.connection || p.status, true)),
    h("div", { class: "row small" },
      h("span", null, `${p.status || "?"}${p.runtime_status && p.runtime_status !== p.status ? " / " + p.runtime_status : ""}`),
      h("span", null, `up ${fmtDur(p.age_s)}`),
      h("span", null, `${fmtUSD(p.cost_so_far_usd)} so far`),
      k.at ? h("span", { class: k.armed === false ? "crit-text" : "" }, `kill switch ${fmtTime(k.at)} (${k.armed === false ? "NOT armed" : "armed"}${k.margin_s !== undefined && k.margin_s !== null ? `, ${fmtDur(k.margin_s)} margin` : ""}; ${k.source || "estimate"})`) : h("span", { class: "muted" }, "kill switch unknown"),
      p.gpu_util ? h("span", null, `GPU ${p.gpu_util.util_pct} % · ${Math.round(p.gpu_util.mem_used_mib / 1024)}/${Math.round(p.gpu_util.mem_total_mib / 1024)} GB`) : h("span", { class: "muted", title: "needs progress level L1 or the live probe" }, "GPU use: n/a (L0)"),
      h("span", { class: "muted" }, `ledger: ${p.ledger || "?"}`)),
    p.session ? h("div", { class: "row small" }, "session ", h("a", { class: "mono", href: `#/runpod/session/${p.session.sid}` }, p.session.sid),
      p.session.take ? h("span", null, p.session.take) : null, p.session.phase ? h("span", { class: "chip" }, p.session.phase) : null) : null,
    p.session && snaps[p.session.sid] ? sessionLive(snaps[p.session.sid])
      : p.session && p.session.progress ? progressBar(p.session.progress, { eta: p.session.eta }) : null,
    flags.length ? h("div", { class: "flags" }, flags.map((f) => pill(FLAG_TEXT[f] || f, "crit"))) : null,
    h("div", { class: "row" },
      p.session ? h("a", { class: "btn small", href: `#/runpod/session/${p.session.sid}` }, "Open session") : null,
      h("span", { class: "spacer" }),
      p.managed ? h("button", { class: "btn small danger", onclick: () => deletePod(app, p) }, "Delete pod…")
        : h("span", { class: "small muted" }, "no actions: created outside dancecap")));
}

export default {
  title: () => "Runpod",
  async mount(root, route, app) {
    let rp = null, sessions = [], snaps = {};
    let showAll = false;
    let fp = null;
    async function load() {
      try {
        const [a, b] = await Promise.all([getx("/api/runpod"), getx("/api/sessions?limit=100")]);
        rp = a.data; sessions = listOf(b.data, "sessions");
        snaps = await activeSnapshots(rp);
      } catch (e) {
        replace(root, h("div", { class: "card crit-text" }, errorText(e)));
        return;
      }
      const f = fingerprint({ rp, sessions, showAll, snaps });
      if (f === fp) return;
      fp = f;
      render();
    }
    function render() {
      const acc = rp.account || {};
      const bud = rp.budget || (app.state && app.state.budget);
      const pods = rp.pods || [];
      const list = showAll ? sessions : sessions.slice(0, 12);
      replace(root,
        h("div", { class: "page-head" }, h("h1", null, "Runpod"), connPill(rp.connection),
          h("span", { class: "muted small" }, rp.checked ? `checked ${fmtAgo(rp.checked)}` : ""),
          rp.message ? h("span", { class: "warn-text small" }, rp.message) : null,
          h("span", { class: "spacer" }),
          h("button", { class: "btn", onclick: async () => {
            try { await post("/api/runpod/refresh", {}); fp = null; await load(); toast("Pod list refreshed.", "good", { timeout: 2500 }); }
            catch (e) { toast(errorText(e), e.status === 429 ? "warn" : "crit"); }
          } }, "Refresh now"),
          orphanPods(pods).length ? h("button", { class: "btn danger", title: "Delete only the dancecap pods flagged orphan (alive, but no session and no runner)",
            onclick: () => cleanupOrphans(app, () => { fp = null; return load(); }) }, `Clean up ${orphanPods(pods).length} orphan pod${orphanPods(pods).length > 1 ? "s" : ""}…`) : null),
        h("div", { class: "grid-main-side" },
          h("div", null,
            h("div", { class: "card" }, budgetGauge(budgetFromState(bud), { title: "Budget: spent, running and projected against the cap" }),
              (bud && bud.messages || []).length ? h("ul", { class: "small warn-text" }, bud.messages.map((m) => h("li", null, m))) : null,
              phaseCosts(bud && bud.phase_costs)),
            h("div", { class: "card" },
              h("div", { class: "card-head" }, h("h3", null, `Pods (${pods.length})`)),
              pods.length ? pods.map((p) => podCard(app, p, snaps)) : h("div", { class: "empty" }, "No pods running."),
              (rp.phantom_accrual || []).length ? h("p", { class: "crit-text small" }, "Phantom accrual (ledger rows open, pod gone): ", rp.phantom_accrual.map((x) => (typeof x === "string" ? x : x.pod_id || x.id || JSON.stringify(x))).join(", ")) : null,
              h("p", { class: "small muted" }, "Read-only by default: the monitor never creates, stops or deletes a pod on its own. A pod is created only by a plan you approve (Plan & run); deleting needs the last 4 characters of the pod id.")),
            h("div", { class: "card" },
              h("div", { class: "card-head" }, h("h3", null, `Sessions (${sessions.length})`),
                sessions.length > 12 ? h("button", { class: "btn small ghost", onclick: () => { showAll = !showAll; fp = null; load(); } }, showAll ? "show recent" : `show all ${sessions.length}`) : null),
              sessions.length ? h("div", { class: "tbl-wrap", style: { maxHeight: "520px" } }, h("table", { class: "tbl compact" },
                h("thead", null, h("tr", null, ["session", "take", "jobs", "state", "origin", "started", "duration", "GPU", "cost"].map((x) => h("th", null, x)))),
                h("tbody", null, list.map((x) => h("tr", { class: "clickable", onclick: () => app.navigate(`#/runpod/session/${x.sid}`) },
                  h("td", { class: "mono small" }, x.sid), h("td", null, x.take || ""), h("td", { class: "small" }, (x.jobs || []).join(", ")),
                  h("td", { title: x.error || "" }, sessionStatePill(x.state)), h("td", { class: "small muted" }, x.origin || ""),
                  h("td", { class: "small", style: { whiteSpace: "nowrap" } }, fmtTime(x.started, true)),
                  // only a live session's duration grows; an ended one without an end time shows none
                  h("td", { class: "small num" }, x.started && (x.ended || ["running", "preparing", "orphaned"].includes(x.state))
                    ? fmtDur((x.ended || Date.now() / 1000) - x.started) : ""),
                  h("td", { class: "small", style: { whiteSpace: "nowrap" } }, (x.gpu || "–").replace("NVIDIA ", "")),
                  h("td", { class: "num" }, x.cost_usd !== null && x.cost_usd !== undefined ? fmtUSD(x.cost_usd) : "–")))))) : h("div", { class: "empty" }, "No sessions yet."))),
          h("div", null,
            h("div", { class: "card" }, h("h3", { class: "section-title" }, "Account"),
              h("div", { class: "kv" },
                h("span", null, "API"), h("span", null, rp.connection === "ok" ? "ok" : rp.connection),
                h("span", null, "balance"), h("span", null, acc.balance_usd !== null && acc.balance_usd !== undefined ? fmtUSD(acc.balance_usd) : "–"),
                h("span", null, "spending now"), h("span", null, acc.spend_per_hr_usd !== null && acc.spend_per_hr_usd !== undefined ? `${fmtUSD(acc.spend_per_hr_usd)}/h` : "–"),
                h("span", { title: "pods `runpodctl pod list` returns now" }, "pods on Runpod"), h("span", null, rp.connection === "ok" ? String(rp.pods_running || 0) : "–"),
                // not the same count: a row whose close was lost stays open until `remote reconcile` sees the pod gone twice
                h("span", { title: "rows of work/runpod/ledger.json without a delete time" }, "open in the ledger"),
                h("span", { class: bud && (bud.pods || []).some((p) => p.phantom_cost != null) ? "warn-text" : "" }, bud && bud.open_pods != null
                  ? `${bud.open_pods}${(bud.pods || []).filter((p) => p.phantom_cost != null).length ? ` (${(bud.pods || []).filter((p) => p.phantom_cost != null).length} past their kill time)` : ""}` : "–"),
                h("span", null, "progress level"), h("span", { title: "L0 parses logs; L1 adds live GPU/upload probes; L2 instruments the stages" }, rp.progress_level || "L0"),
                h("span", null, "cap"), h("span", null, bud ? `${fmtUSD(bud.cap)} (reserve ${fmtUSD(bud.reserve)})` : "–"),
                h("span", null, "remaining"), h("span", null, bud ? fmtUSD(bud.remaining) : "–"))),
            h("div", { class: "card" }, h("h3", { class: "section-title" }, "How costs are counted"),
              h("ul", { class: "small", style: { paddingLeft: "18px", margin: 0, color: "var(--text-2)" } },
                h("li", null, "Spent comes from the ledger (work/runpod/ledger.json); open rows keep accruing."),
                h("li", null, "Expected (p50) and pessimistic (p90) add price × ETA for running and planned pods."),
                h("li", null, "The worst case is price × time until each running pod's kill switch fires."),
                h("li", null, "Upload time is billed GPU time: the pod exists while the bundle uploads."))))));
    }
    await load();
    return { refresh: load };
  },
};
