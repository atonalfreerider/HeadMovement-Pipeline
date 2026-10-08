// Shared actions: Plan & run (the only path to a paid pod: plan -> explicit approval -> run), Export to Unity,
// Narrate moves, Open Unity / Load capture, consent, stop a run, reveal in Explorer.
import { h, replace, fmtDur, fmtUSD, fmtEta, fmtCostBand, fmtBytes, debounce } from "../util.js";
import { get, post, patch, seg } from "../api.js";
import { openDialog, toast, errorText, confirmDialog } from "../components/dialog.js";
import { statusGlyph, whereBadge } from "../components/status.js";
import { budgetGauge, budgetFromState } from "../components/progress.js";
import { CLASS_LABEL } from "../components/cameraType.js";

const ACTION_LABEL = { run: "run", rerun_stale: "re-run (stale)", skip_done: "skip (done)", blocked: "needs you", "n/a": "n/a",
  pause: "pauses for you" };
const CT_STATE = { confirmed: "yours", suggested: "suggestion, not confirmed", unset: "not set", conflict: "conflict" };

/** The camera model each video gets in this plan (the effective class: your answer; an unconfirmed suggestion is solved
 * as handheld + zoom), with what the class changes in the camera stages. */
function cameraTypesBlock(p, take, dlg) {
  const cts = p.camera_types || [];
  if (!cts.length) return null;
  return h("div", { class: "card", style: { marginTop: "10px", padding: "8px 12px" } },
    h("div", { class: "row small" }, h("strong", null, "Camera types this plan applies"),
      h("span", { class: "muted" }, "(the camera stages solve each video with this model)"),
      h("span", { class: "spacer" }),
      h("a", { class: "btn small ghost", href: `#/take/${take}/videos`, onclick: () => dlg.close() }, "Review videos")),
    h("table", { class: "tbl compact", style: { marginTop: "4px" } },
      h("tbody", null, cts.map((c) => h("tr", null,
        h("td", { class: "mono" }, c.cam),
        h("td", null, h("strong", null, CLASS_LABEL[c.class] || c.class)),
        h("td", { class: c.state === "confirmed" ? "good-text small" : "warn-text small" }, CT_STATE[c.state] || c.state,
          c.state !== "confirmed" && c.suggested && c.suggested.mount ? ` (suggested ${c.suggested.mount}${c.suggested.zoom ? " + zoom" : ""}${c.suggested.confidence != null ? `, ${Math.round(c.suggested.confidence * 100)} %` : ""})` : ""),
        h("td", { class: "muted small" }, (c.effects || []).join("; ")))))));
}

function stepCameras(s) {
  const ct = s.camera_types;
  if (!ct) return null;
  return h("div", { class: "small", style: { marginTop: "2px" } }, Object.entries(ct).map(([k, cams]) =>
    h("span", { class: "chip", title: CLASS_LABEL[k] || k, style: { marginRight: "4px" } }, `${CLASS_LABEL[k] || k}: ${cams.join(" ")}`)));
}

function runLink(run) {
  return h("a", { href: `#/runs/${run.id}` }, run.id);
}

// ------------------------------------------------------------------------------------------------ plan & run

/**
 * planDialog(app, take, {target, scope, mode}) — POST /plan has no side effects; the Run button needs an explicit
 * approval checkbox whenever the plan creates pods, and the server re-checks hash, budget and consent.
 */
export async function planDialog(app, take, init = {}) {
  let snap;
  try { snap = await get(`/api/takes/${seg(take)}/pipeline`); } catch (e) { toast(errorText(e), "crit"); return; }
  const stages = (snap.stages || []).filter((s) => s.in_variant !== false);
  const req = {
    target: init.target || (stages.find((s) => s.id === "demo_export") ? "demo_export" : stages[stages.length - 1].id),
    scope: init.scope || "upto", mode: init.mode || "resume", include_optional: [], options: {},
  };
  let plan = null;
  let showSkipped = false;
  if (init.whatIf) req.options.assume_reviews = true;
  let approve = false;
  let approveOver = false;
  const runnable = stages.filter((s) => s.runs_on !== "human" && s.implemented !== false);
  const optional = stages.filter((s) => s.optional && s.implemented !== false && s.runs_on !== "human");

  const targetSel = h("select", { "aria-label": "Target stage" },
    runnable.map((s) => h("option", { value: s.id, selected: s.id === req.target }, `${s.id} — ${s.name}`)));
  const scopeSel = h("select", { "aria-label": "Scope" },
    [["upto", "it and every not-done stage before it"], ["only", "only this stage"], ["all", "the whole pipeline"]]
      .map(([v, l]) => h("option", { value: v, selected: v === req.scope }, l)));
  const modeSel = h("select", { "aria-label": "Mode" },
    [["resume", "resume (skip fresh outputs)"], ["force", "force (re-run even if done)"]]
      .map(([v, l]) => h("option", { value: v, selected: v === req.mode }, l)));
  const capIn = h("input", { type: "text", class: "mono", placeholder: "e.g. 07_LarissaKadu", autocomplete: "off", style: { width: "180px" } });
  suggestCapture(take).then((c) => { if (c && !capIn.value) { capIn.value = c; req.options = { ...req.options, capture: c }; replan(); } });
  const whatIfBox = h("input", { type: "checkbox", checked: !!req.options.assume_reviews });
  whatIfBox.addEventListener("change", () => { req.options = { ...req.options, assume_reviews: whatIfBox.checked || undefined }; replan(); });
  const optBoxes = optional.map((s) => {
    const cb = h("input", { type: "checkbox", value: s.id });
    cb.addEventListener("change", () => { req.include_optional = optBoxes.filter((b) => b.checked).map((b) => b.value); replan(); });
    return cb;
  });
  const out = h("div", { class: "plan-out" }, h("div", { class: "loading" }, "planning…"));
  const body = h("div", { class: "plan" },
    h("div", { class: "form-grid" },
      h("label", { class: "field" }, h("span", null, "Run up to"), targetSel),
      h("label", { class: "field" }, h("span", null, "Scope"), scopeSel),
      h("label", { class: "field" }, h("span", null, "Mode"), modeSel),
      h("label", { class: "field", title: "The Unity capture folder demo_export writes (StreamingAssets/<name>); a new take has none yet" },
        h("span", null, "Unity capture name"), capIn)),
    optional.length ? h("div", { class: "row small", style: { marginTop: "8px" } }, h("span", { class: "muted" }, "Optional stages:"),
      optional.map((s, i) => h("label", { class: "chk" }, optBoxes[i], s.id))) : null,
    h("div", { class: "row small", style: { marginTop: "8px" } }, h("label", { class: "chk", title: "Dry-run only: the human steps (labels review, narration, review gates) are assumed done, so you see the whole order, wall time and pod cost of a new capture before any review exists" },
      whatIfBox, h("strong", null, "What-if"), " — preview the whole pipeline as if my reviews were done (dry-run only)")),
    h("hr"), out);

  const dlg = openDialog({ title: `Plan & run — ${take}`, body, wide: true, actions: [] });
  capIn.addEventListener("input", () => {
    const v = capIn.value.trim();
    const okName = /^[A-Za-z0-9_-]{1,64}$/.test(v);
    capIn.classList.toggle("invalid", !!v && !okName);
    if (!v || okName) { req.options = { ...req.options, capture: v || undefined }; replan(); }
  });
  for (const [el, k] of [[targetSel, "target"], [scopeSel, "scope"], [modeSel, "mode"]]) {
    el.addEventListener("change", () => { req[k] = el.value; replan(); });
  }

  // Only the newest plan request may draw (responses can arrive out of order), and between a change of the inputs
  // and that plan's answer nothing can be started: the plan on screen no longer matches the selects.
  let planSeq = 0;
  let pending = true;
  const replanSoon = debounce(doPlan, 150);
  function replan() {
    pending = true;
    if (plan) setActs();
    replanSoon();
  }
  async function doPlan() {
    const seq = ++planSeq;
    pending = true;
    replace(out, h("div", { class: "loading" }, "planning…"));
    try {
      const p = await post(`/api/takes/${seg(take)}/plan`, { ...req, include_optional: req.include_optional.slice() });
      if (seq !== planSeq) return;
      plan = p;
      pending = false;
      approve = false; approveOver = false;
      draw();
    } catch (e) {
      if (seq !== planSeq) return;
      pending = false;
      plan = null;
      replace(out, h("div", { class: "crit-text" }, errorText(e)));
      dlg.setActions([{ label: "Close" }]);
    }
  }

  function draw() {
    const p = plan;
    const steps = p.steps || [];
    const todo = steps.filter((s) => s.action !== "skip_done" && s.action !== "n/a");
    const shown = showSkipped ? steps : todo;
    const sessions = p.sessions || [];
    const maxCost = sessions.reduce((a, s) => a + (s.max_cost_usd || 0), 0);
    const maxMin = sessions.reduce((a, s) => Math.max(a, s.max_minutes || 0), 0);
    const gate = (p.budget || {}).gate || "ok";
    const consentOk = !sessions.length || (p.consent || {}).ok !== false;
    const band = p.camera_type_band || {};
    replace(out,
      h("div", { class: "row" },
        h("strong", null, `${take} → ${p.request ? p.request.target : req.target}`),
        h("span", { class: "muted" }, `(${req.mode}, ${req.scope})`),
        h("span", { class: "spacer" }),
        h("span", { class: "chip mono", title: p.plan_hash }, "plan " + String(p.plan_hash || "").replace("sha1:", "").slice(0, 8)),
        h("span", { class: "muted small" }, `valid ${Math.round((p.valid_for_s || 600) / 60)} min`)),
      h("div", { class: "metrics", style: { margin: "10px 0" } },
        metric("Stages to run", String(todo.filter((s) => s.action === "run" || s.action === "rerun_stale").length),
          `${steps.length - todo.length} already done`),
        metric("Wall time", fmtEta({ p50_s: p.totals && p.totals.wall_s && p.totals.wall_s.p50, p90_s: p.totals && p.totals.wall_s && p.totals.wall_s.p90 }),
          p.totals && p.totals.local_s ? `${fmtDur(p.totals.local_s)} on this PC` : ""),
        metric("Pod cost", fmtCostBand(p.totals && p.totals.pod_cost_usd), sessions.length ? `${sessions.length} pod session${sessions.length > 1 ? "s" : ""}` : "no pods"),
        metric("Budget after (p90)", fmtUSD(p.budget && p.budget.after_p90), `cap ${fmtUSD(p.budget && p.budget.cap, 0)}, reserve ${fmtUSD(p.budget && p.budget.reserve, 0)}`)),
      h("div", { class: "tbl-wrap", style: { maxHeight: "300px" } },
        h("table", { class: "tbl compact" },
          h("thead", null, h("tr", null, ["", "stage", "where", "action", "ETA p50 / p90", "cost p50 / p90", "session", "reason"].map((x) => h("th", null, x)))),
          h("tbody", null, shown.map((s) => h("tr", null,
            h("td", null, statusGlyph(s.status)),
            h("td", null, h("strong", null, s.stage)),
            h("td", null, whereBadge(s.runs_on)),
            h("td", { class: s.action === "blocked" ? "needs" : "" }, ACTION_LABEL[s.action] || s.action),
            h("td", { class: "num" }, s.action === "skip_done" ? "–" : fmtEta(s.est)),
            h("td", { class: "num" }, s.action === "skip_done" ? "–" : fmtCostBand(s.cost)),
            h("td", null, s.session ? h("span", { class: "chip session" }, s.session) : ""),
            h("td", { class: "muted small" }, s.reason || "", stepCameras(s))))))),
      h("div", { class: "row small", style: { marginTop: "6px" } },
        h("label", { class: "chk" }, (() => { const cb = h("input", { type: "checkbox", checked: showSkipped });
          cb.addEventListener("change", () => { showSkipped = cb.checked; draw(); }); return cb; })(),
          `show done stages (${steps.length - todo.length})`)),
      p.what_if ? h("p", { class: "warn-text" }, "What-if plan: dry-run only. Your reviews are assumed done so the whole order, time and cost show; a real run stops at the first of: ", (p.halts_at || []).join(", ") || "(none)")
        : (p.halts_at || []).length ? h("p", { class: "needs" }, `The run pauses for you at: ${(p.halts_at || []).join(", ")}`) : null,
      cameraTypesBlock(p, take, dlg),
      sessions.length ? [h("h3", { class: "section-title", style: { marginTop: "14px" } }, "Pod sessions (paid GPU)"),
        h("div", { class: "tbl-wrap" }, h("table", { class: "tbl compact" },
          h("thead", null, h("tr", null, ["session", "stages", "GPU", "cloud", "$/h", "upload", "time p50 / p90", "cost p50 / p90", "kill switch", "max cost"].map((x) => h("th", null, x)))),
          h("tbody", null, sessions.map((s) => h("tr", null,
            h("td", null, h("span", { class: "chip session" }, s.id)),
            h("td", { class: "small" }, (s.stages || []).join(", ")),
            h("td", null, s.gpu || "–"), h("td", null, (s.cloud || "").toLowerCase()),
            h("td", { class: "num" }, fmtUSD(s.price_per_hr)),
            h("td", { class: "num", title: s.overhead_min ? `upload p50 ${Math.round(s.overhead_min.upload_p50_min)} min / p90 ${Math.round(s.overhead_min.upload_p90_min)} min (billed: the pod exists during the upload)` : "" },
              fmtBytes(s.upload_bytes)),
            h("td", { class: "num" }, s.est_min ? `${fmtDur(s.est_min.p50 * 60)} / ${fmtDur(s.est_min.p90 * 60, { short: true })}` : "–"),
            h("td", { class: "num" }, fmtCostBand(s.cost_usd)),
            h("td", { class: "num" }, s.max_minutes ? `${s.max_minutes} min` : "–"),
            h("td", { class: "num" }, fmtUSD(s.max_cost_usd)))))))] : null,
      h("div", { style: { marginTop: "14px" } }, budgetGauge(budgetFromState(app.state && app.state.budget, p), { title: "Budget with this plan" })),
      sessions.length ? h("div", { class: ["row", consentOk ? "good-text" : "crit-text"], style: { marginTop: "10px" } },
        consentOk ? "✓ cloud_processing consent recorded for every dancer"
          : ["✕ cloud_processing consent is not recorded as yes for every dancer: pod stages cannot start. ",
            h("button", { class: "btn small", onclick: async () => { await consentDialog(app, take); replan(); } }, "Record consent…")]) : null,
      band.unconfirmed && band.unconfirmed.length ? h("div", { class: "row warn-text", style: { marginTop: "6px" } },
        `! ${band.unconfirmed.length} camera type${band.unconfirmed.length > 1 ? "s" : ""} unconfirmed (${band.unconfirmed.join(", ")}): p90 priced as handheld + zoom`,
        band.p90_extra_pod_min ? ` (+${Math.round(band.p90_extra_pod_min)} pod-min, +${fmtUSD(band.p90_extra_usd)})` : "",
        h("a", { class: "btn small", href: `#/take/${take}/videos`, onclick: () => dlg.close() }, "Review videos")) : null,
      (p.warnings || []).length ? h("ul", { class: "small warn-text", style: { margin: "8px 0 0", paddingLeft: "18px" } },
        p.warnings.map((w) => h("li", null, w))) : null,
      sessions.length && gate !== "over_cap" && !p.what_if ? h("div", { class: "card", style: { marginTop: "12px", borderColor: "var(--accent)" } },
        h("label", { class: "chk" }, (() => { const cb = h("input", { type: "checkbox", checked: approve });
          cb.addEventListener("change", () => { approve = cb.checked; setActs(); }); return cb; })(),
          h("strong", null, `I approve creating ${sessions.length} paid pod${sessions.length > 1 ? "s" : ""} `),
          `(at most ${fmtUSD(maxCost)} in total; each pod deletes itself after its kill switch, ${maxMin} min at the longest)`),
        gate === "over_reserve" ? h("label", { class: "chk", style: { marginTop: "6px" } }, (() => {
          const cb = h("input", { type: "checkbox", checked: approveOver });
          cb.addEventListener("change", () => { approveOver = cb.checked; setActs(); }); return cb; })(),
          h("span", { class: "warn-text" }, "I accept that the p90 goes over cap - reserve")) : null) : null,
      gate === "over_cap" ? h("div", { class: "card crit-text", style: { marginTop: "12px" } },
        "✕ The p90 of this plan would exceed the cap. Confirm camera types, choose fewer stages, or raise the cap (remote.py) first.") : null);
    setActs();
  }

  function setActs() {
    const p = plan;
    const sessions = p.sessions || [];
    const gate = (p.budget || {}).gate || "ok";
    const todo = (p.steps || []).filter((s) => s.action === "run" || s.action === "rerun_stale");
    const consentOk = !sessions.length || (p.consent || {}).ok !== false;
    let why = null;
    if (pending) why = "re-planning…";
    else if (p.what_if) why = "what-if plans are dry-run only";
    else if (!todo.length) why = "nothing to run: every stage of this plan is done";
    else if (sessions.length && gate === "over_cap") why = "over the cap";
    else if (sessions.length && !consentOk) why = "cloud_processing consent missing";
    else if (sessions.length && !approve) why = "tick the approval to create pods";
    else if (sessions.length && gate === "over_reserve" && !approveOver) why = "tick the over-reserve approval";
    dlg.setActions([
      why ? { node: h("span", { class: "muted small", style: { marginRight: "auto" } }, why) } : { node: h("span", { style: { marginRight: "auto" } }) },
      { label: "Cancel" },
      { label: p.what_if ? "Dry-run the whole pipeline" : sessions.length ? "Dry-run (bundle, no pod)" : "Dry-run", disabled: pending || !todo.length,
        title: "Runs the plan's dry-run steps: builds bundles and estimates, never creates a pod", onClick: () => start(true) },
      { label: sessions.length ? `Run (creates ${sessions.length} pod${sessions.length > 1 ? "s" : ""})` : "Run", kind: "primary",
        disabled: !!why, onClick: () => start(false) },
    ]);
  }

  async function start(dry) {
    if (pending || !plan) return false;
    const p = plan;
    const sessions = p.sessions || [];
    const bodyReq = { plan_hash: p.plan_hash, dry_run: dry };
    if (!dry && sessions.length) {
      bodyReq.approve = { pods: sessions.length, max_cost_usd: sessions.reduce((a, s) => a + (s.max_cost_usd || 0), 0),
        over_reserve: approveOver || undefined };
    }
    try {
      const res = await post(`/api/takes/${seg(take)}/runs`, bodyReq);
      toast([dry ? "Dry-run started: " : "Run started: ", runLink(res.run)], "good");
      app.refreshNow();
      return true;
    } catch (e) {
      if (e.status === 412 && e.details && e.details.plan) {
        plan = e.details.plan; approve = false; approveOver = false; draw();
        toast("The plan changed (files or prices moved): review it again.", "warn");
        return false;
      }
      toast(errorText(e), "crit");
      return false;
    }
  }

  doPlan();
}

/** The take's last Unity capture, else the next free "NN_<TakeInCamelCase>" (NN after the highest in StreamingAssets). */
async function suggestCapture(take) {
  try {
    const d = await get(`/api/takes/${seg(take)}`);
    if (d.exports && d.exports[0] && d.exports[0].capture) return d.exports[0].capture;
  } catch { /* fall through */ }
  let nn = 1;
  try {
    const u = await get("/api/unity");
    for (const c of u.captures || []) { const m = /^(\d+)_/.exec(c); if (m) nn = Math.max(nn, parseInt(m[1], 10) + 1); }
  } catch { /* no Unity info */ }
  const camel = String(take).split(/[^A-Za-z0-9]+/).filter(Boolean).map((x) => x[0].toUpperCase() + x.slice(1)).join("");
  return `${String(nn).padStart(2, "0")}_${camel}`.slice(0, 64);
}

function metric(label, value, sub) {
  return h("div", { class: "metric" }, h("div", { class: "m-label" }, label), h("div", { class: "m-value" }, value),
    sub ? h("div", { class: "m-sub" }, sub) : null);
}

// ------------------------------------------------------------------------------------------------ export

export async function exportDialog(app, take) {
  let detail = null;
  try { detail = await get(`/api/takes/${seg(take)}`); } catch { /* defaults below */ }
  const last = detail && detail.exports && detail.exports[0] ? detail.exports[0].capture : "";
  const cap = h("input", { type: "text", value: last, placeholder: "e.g. 05_LarissaKadu", class: "mono", autocomplete: "off" });
  const smplx = h("select", null, h("option", { value: "" }, "default (the take's [export] motion)"),
    h("option", { value: "smplx" }, "smplx (raw body fit)"));
  const audio = h("select", null, ["auto", "studio", "phone"].map((v) => h("option", { value: v }, v)));
  const room = h("input", { type: "checkbox" });
  const lock = app.state && app.state.locks && app.state.locks.unity;
  const override = h("input", { type: "checkbox" });
  openDialog({
    title: `Export to Unity — ${take}`,
    body: h("div", { class: "wiz-body" },
      h("p", { class: "muted" }, "Runs ", h("code", null, "python -m dancecap.demo_export"),
        " on this PC and writes the capture into the Unity project's StreamingAssets (no dancer media goes into the public repo)."),
      h("div", { class: "form-grid" },
        h("label", { class: "field" }, h("span", null, "Capture name"), cap),
        h("label", { class: "field" }, h("span", null, "Body fit"), smplx),
        h("label", { class: "field" }, h("span", null, "Audio"), audio)),
      h("label", { class: "chk" }, room, "include the room splat"),
      lock && lock.held ? h("div", { class: "card warn-text" },
        h("div", null, "! Unity work is in progress (work/unity.lock): ", h("span", { class: "muted" }, lock.owner || "")),
        h("label", { class: "chk", style: { marginTop: "6px" } }, override, "export anyway (may disturb that work)")) : null),
    actions: [{ label: "Cancel" }, { label: "Dry run", title: "Writes nothing: lists the inputs found / missing, the folders it would replace and the triggers it would carry", onClick: async () => {
      if (!/^[A-Za-z0-9_-]{1,64}$/.test(cap.value.trim())) { cap.classList.add("invalid"); toast("Capture name: letters, digits, _ and -", "warn"); return false; }
      try {
        const res = await post(`/api/takes/${seg(take)}/export-dry-run`, {
          capture: cap.value.trim(), smplx_dir: smplx.value || undefined, audio: audio.value, room: room.checked || undefined });
        toast(["Export dry run (nothing is written): ", runLink(res.run)], "good");
        app.refreshNow();
        app.navigate(`#/runs/${res.run.id}`);
      } catch (e) {
        toast(errorText(e), "crit");
        return false;
      }
    } }, { label: "Export", kind: "primary", onClick: async () => {
      if (!/^[A-Za-z0-9_-]{1,64}$/.test(cap.value.trim())) { cap.classList.add("invalid"); toast("Capture name: letters, digits, _ and -", "warn"); return false; }
      try {
        const res = await post(`/api/takes/${seg(take)}/export`, {
          capture: cap.value.trim(), smplx_dir: smplx.value || undefined, audio: audio.value, room: room.checked || undefined,
          override_lock: override.checked || undefined });
        toast(["Export started: ", runLink(res.run)], "good");
        app.refreshNow();
      } catch (e) {
        toast(errorText(e), e.status === 423 ? "warn" : "crit");
        return false;
      }
    } }],
  });
}

// ------------------------------------------------------------------------------------------------ narration

export async function narrate(app, take, camera) {
  let res;
  try {
    res = await post(`/api/takes/${seg(take)}/narrate`, camera ? { camera } : {});
  } catch (e) { toast(errorText(e), "crit"); return; }
  const run = res.run;
  const t = toast(res.reused ? "The narration tool is already running for this take: opening its window…"
    : "Starting the narration tool (it opens in its own window)…", "info", { timeout: 0 });
  const t0 = Date.now();
  while (Date.now() - t0 < 45000) {
    let r = run;
    try { r = await get(`/api/runs/${seg(run.id)}`); } catch { /* keep waiting */ }
    if (r.url) {
      try { await post("/api/shell/open", { tool_run: run.id }); t.remove(); toast("Narration tool opened.", "good"); }
      catch (e) { t.remove(); toast(errorText(e), "warn"); }
      app.refreshNow();
      return;
    }
    if (["failed", "lost", "done", "stopped"].includes(r.state)) {
      t.remove();
      toast(["The narration tool exited (", r.state, "): ", runLink(r)], "crit");
      return;
    }
    await new Promise((ok) => setTimeout(ok, 1000));
  }
  t.remove();
  toast(["The narration tool is still starting: open it later from ", runLink(run)], "warn");
}

// ------------------------------------------------------------------------------------------------ unity

export async function openUnity(app) {
  try {
    const r = await post("/api/unity/open", {});
    toast(r.running ? "Unity is already open." : "Starting the Unity editor…", "good");
  } catch (e) { toast(errorText(e), "crit"); }
}

export async function loadCaptureDialog(app, take) {
  let u;
  try { u = await get("/api/unity"); } catch (e) { toast(errorText(e), "crit"); return; }
  const sel = h("select", null, (u.captures || []).map((c) => h("option", { value: c }, c)));
  openDialog({
    title: "Load a capture in Unity",
    body: h("div", { class: "wiz-body" },
      h("p", { class: "muted" }, "Sends hm_load to the running editor through the Unity CLI (Play mode only)."),
      u.running ? null : h("p", { class: "warn-text" }, "! Unity does not look open."),
      (u.captures || []).length ? h("label", { class: "field" }, h("span", null, "Capture"), sel)
        : h("p", null, "No exported captures in StreamingAssets yet.")),
    actions: [{ label: "Cancel" }, { label: "Load", kind: "primary", disabled: !(u.captures || []).length, onClick: async () => {
      try { const r = await post("/api/unity/load", { capture: sel.value }); toast(`Loaded (rc ${r.rc}). ${r.output || ""}`, "good"); }
      catch (e) { toast(errorText(e), "crit"); return false; }
    } }],
  });
}

// ------------------------------------------------------------------------------------------------ consent

const SCOPES = [
  ["capture", "capture"], ["fit_3d", "3D fit"], ["ml_analysis", "ML analysis"], ["cloud_processing", "cloud (Runpod)"],
  ["name_shown", "name shown"], ["level_shown", "level shown"], ["publication", "publication"], ["public_repo", "public repo"],
];

export async function consentDialog(app, take) {
  let d;
  try { d = await get(`/api/takes/${seg(take)}`); } catch (e) { toast(errorText(e), "crit"); return; }
  const roles = Object.keys(d.dancers || { lead: 1, follow: 1 });
  const c = JSON.parse(JSON.stringify(d.consent || {}));
  for (const r of roles) c[r] = c[r] || {};
  const cycle = (v) => (v === true ? false : v === false ? "?" : true);
  const label = (v) => (v === true ? "yes" : v === false ? "no" : "?");
  const btn = (role, k) => {
    const b = h("button", { type: "button", class: ["tri", c[role][k] === true ? "yes" : c[role][k] === false ? "no" : ""] }, label(c[role][k]));
    b.addEventListener("click", () => {
      c[role][k] = cycle(c[role][k]);
      b.textContent = label(c[role][k]);
      b.className = "tri " + (c[role][k] === true ? "yes" : c[role][k] === false ? "no" : "");
    });
    return b;
  };
  const rights = h("input", { type: "text", value: c.source_rights && c.source_rights !== "?" ? c.source_rights : "", placeholder: "who owns the footage / permission" });
  const bys = h("input", { type: "text", value: c.bystanders && c.bystanders !== "?" ? c.bystanders : "", placeholder: "bystanders visible? blurred?" });
  return new Promise((resolve) => {
    openDialog({
      title: `Consent — ${take}`, wide: true,
      body: h("div", { class: "wiz-body" },
        h("p", { class: "muted" }, "Recorded in the take TOML [consent]. cloud_processing = yes is required before any Runpod stage; \"?\" blocks the stages that need that scope. Click to cycle yes → no → ?."),
        h("table", { class: "tbl compact consent-tbl" },
          h("thead", null, h("tr", null, h("th", null, "dancer"), SCOPES.map(([, l]) => h("th", null, l)))),
          h("tbody", null, roles.map((r) => h("tr", null,
            h("td", null, h("strong", null, r), " ", h("span", { class: "muted" }, (d.dancers[r] || {}).name || "")),
            SCOPES.map(([k]) => h("td", null, btn(r, k))))))),
        h("div", { class: "form-grid" },
          h("label", { class: "field" }, h("span", null, "Source rights"), rights),
          h("label", { class: "field" }, h("span", null, "Bystanders"), bys))),
      actions: [{ label: "Cancel", onClick: () => resolve(false) }, { label: "Save", kind: "primary", onClick: async () => {
        const consent = {};
        for (const r of roles) consent[r] = c[r];
        consent.source_rights = rights.value.trim() || "?";
        consent.bystanders = bys.value.trim() || "?";
        try {
          await patch(`/api/takes/${seg(take)}`, { consent }, { ifMatch: d.toml_etag });
          toast("Consent saved to the take TOML.", "good");
          app.refreshNow();
          resolve(true);
        } catch (e) {
          toast(e.status === 412 ? "The take TOML changed meanwhile: reopen the dialog." : errorText(e), "crit");
          return false;
        }
      } }],
      onClose: () => resolve(false),
    });
  });
}

// ------------------------------------------------------------------------------------------------ runs

export async function stopRun(app, run) {
  const modes = run.stop_modes || [];
  if (!modes.length) { toast("This run cannot be stopped from the monitor.", "warn"); return; }
  const mode = modes.includes("after_step") ? "after_step" : modes[0];
  const ok = await confirmDialog(`Stop ${run.id}?`,
    [h("p", null, mode === "after_step" ? "The run stops before its next stage (the current one finishes)."
      : "The run is interrupted now; its cleanup (finally blocks) still runs."),
    modes.length > 1 ? h("p", { class: "muted" }, "Use the run page to choose another stop mode.") : null],
    { okLabel: mode === "after_step" ? "Stop after this step" : "Stop now", danger: true });
  if (!ok) return;
  try { await post(`/api/runs/${seg(run.id)}/stop`, { mode }); toast(`Stop requested (${mode}).`, "good"); app.refreshNow(); }
  catch (e) { toast(errorText(e), "crit"); }
}

export async function reveal(take, what, run) {
  try { await post("/api/shell/reveal", { take, what, run }); }
  catch (e) { toast(errorText(e), "warn"); }
}

