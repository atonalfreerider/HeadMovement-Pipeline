// Camera type input (PIPELINE_MONITOR.md 6.4 / 7.3 CameraType): the user's three answers per video:
// mount = Handheld | Tripod (exclusive), zoom = No | Yes (independent: a tripod can zoom). Pre-filled from the
// automatic suggestion (hatched) which the user confirms; the user always wins over the data, and a data conflict
// is shown with its evidence, never applied silently.
import { h, fmtTime, fmtUSD } from "../util.js";

export const CLASS_LABEL = {
  tripod: "Tripod, fixed focal",
  tripod_zoom: "Tripod + zoom",
  handheld: "Handheld, fixed focal",
  handheld_zoom: "Handheld + zoom",
};

export const CLASS_SHORT = { tripod: "T", tripod_zoom: "TZ", handheld: "H", handheld_zoom: "HZ" };

export function classOf(mount, zoom) {
  if (!mount || zoom === null || zoom === undefined) return null;
  return mount + (zoom ? "_zoom" : "");
}

/** What the pipeline does with each class (fallback when the registry text is not loaded). */
export const CLASS_EFFECT = {
  tripod: "one rigid pose for the whole video (keyframe localisation, ~10 % of the per-frame work); one intrinsic set; a constant camera track for sync and the dancers",
  tripod_zoom: "fixed camera centre; per-frame rotation + focal; no zoom-depth ambiguity",
  handheld: "per-frame 6-dof pose with the video's one focal + lens distortion (no focal race)",
  handheld_zoom: "per-frame pose + focal, log-focal smoothing, zoom-depth guard (today's general solver)",
};

const EVIDENCE = [
  ["parallax_px", "background parallax", (v) => `${v.toFixed(2)} px`],
  ["jitter_hf", "hand jitter (2-12 Hz)", (v) => v.toFixed(3)],
  ["zoom_scale", "scale drift", (v) => `${((v - 1) * 100).toFixed(1)} %`],
  ["centre_p95_m", "centre moves (p95)", (v) => `${(v * 100).toFixed(0)} cm`],
  ["axis_p95_deg", "axis turns (p95)", (v) => `${v.toFixed(1)}°`],
  ["angvel_p50_dps", "angular speed (p50)", (v) => `${v.toFixed(1)}°/s`],
  ["focal_ratio", "focal range", (v) => `×${v.toFixed(2)}`],
];

const BY = {
  probe_a: "before calibration: background motion in the video",
  probe_b: "after calibration: the solved camera track",
};

function seg(name, options, value, { provisional, disabled, onPick }) {
  return h("div", { class: ["segmented", provisional ? "provisional" : ""], role: "radiogroup", "aria-label": name },
    options.map(([v, label, key]) => h("button", {
      type: "button", role: "radio", "aria-checked": String(value === v),
      class: ["seg-btn", value === v ? "on" : ""], disabled,
      title: key ? `${label} (key ${key})` : label,
      onclick: () => onPick(v),
    }, label)));
}

/** Small read-only chip, e.g. in tables: "H", "TZ", "?" + state class. */
export function cameraTypeChip(t) {
  if (!t || t.state === "unset") {
    return h("span", { class: "ct-chip ct-unset", title: "not set: the general model (handheld + zoom) is used" }, "?");
  }
  if (t.state === "suggested" && t.suggestion) {
    const sc = classOf(t.suggestion.mount, t.suggestion.zoom);
    return h("span", { class: "ct-chip ct-suggested", title: `suggested ${CLASS_LABEL[sc] || sc} — not confirmed yet` }, (CLASS_SHORT[sc] || "?") + "?");
  }
  const cls = classOf(t.mount, t.zoom) || (t.effective && t.effective.class);
  const label = CLASS_LABEL[cls] || cls;
  return h("span", { class: ["ct-chip", "ct-" + t.state], title: `${label} (${t.state === "conflict" ? "conflict with the data" : "confirmed"})` },
    CLASS_SHORT[cls] || "?", t.state === "conflict" ? " !" : "");
}

export function evidenceList(sugg) {
  if (!sugg || !sugg.evidence) return null;
  const ev = sugg.evidence;
  const items = [];
  for (const [k, label, f] of EVIDENCE) {
    const v = ev[k];
    if (v === null || v === undefined || !isFinite(v)) continue;
    items.push(h("span", { class: "ev" }, h("span", { class: "muted" }, label + " "), f(v)));
  }
  if (ev.mount_judgeable === false) items.push(h("span", { class: "ev muted" }, "mount: centre fixed by the attach model"));
  if (ev.zoom_judgeable === false) items.push(h("span", { class: "ev muted" }, "zoom: focal fixed by the attach model (from the video)"));
  if (ev.mount_lean === "tripod" && sugg.mount === "handheld") items.push(h("span", { class: "ev muted" }, "very still (leans tripod; below the confidence bar)"));
  const zev = ev.zoom_events || sugg.zoom_events || [];
  if (zev.length) items.push(h("span", { class: "ev" }, h("span", { class: "muted" }, "zoom events "), String(zev.length)));
  if (sugg.pans) items.push(h("span", { class: "ev" }, "pans detected"));
  return items.length ? h("div", { class: "evidence" }, items) : null;
}

/**
 * cameraTypeControl({type, onSave(mount, zoom) -> Promise, onAccept() -> Promise, classes, compact})
 * Returns {el, pick(field, value), confirm(), toggleZoom()} so the Video Review keys (H/T/Z/A/Enter) can drive it.
 */
export function cameraTypeControl({ type, onSave, onAccept, classes = null, compact = false, disabled = false }) {
  const t = type || { state: "unset", effective: { mount: "handheld", zoom: true, class: "handheld_zoom" } };
  const sugg = t.suggestion;
  let draft;
  if (t.state === "confirmed" || t.state === "conflict") draft = { mount: t.mount, zoom: t.zoom };
  else if (t.state === "suggested" && sugg) draft = { mount: sugg.mount, zoom: sugg.zoom };
  else draft = { mount: t.mount ?? null, zoom: t.zoom ?? null };
  const provisional = t.state === "suggested" || t.state === "unset";
  const answered = t.state === "confirmed" || t.state === "conflict";
  const touched = new Set(); // a pre-filled (suggested) field is saved only by Confirm or once both were picked here
  let saving = false;
  const el = h("div", { class: ["ct", "ct-state-" + t.state, compact ? "compact" : ""] });

  async function save(m, z) {
    if (saving || disabled) return;
    saving = true;
    el.classList.add("saving");
    try { await onSave(m, z); } finally { saving = false; el.classList.remove("saving"); }
  }
  const complete = () => draft.mount && draft.zoom !== null && draft.zoom !== undefined;
  function pick(field, value) {
    if (disabled) return;
    draft = { ...draft, [field]: value };
    touched.add(field);
    // an answer the user already gave changes field by field; a suggestion is saved only when both fields were
    // picked in this view (or by Confirm) - one click must not confirm the pre-filled other field
    if (complete() && (answered || touched.size === 2)) save(draft.mount, draft.zoom);
    else render();
  }
  async function confirm() {
    const asSuggested = sugg && draft.mount === sugg.mount && draft.zoom === sugg.zoom;
    if (t.state === "suggested" && asSuggested && onAccept) { saving = true; try { await onAccept(); } finally { saving = false; } return; }
    if (complete()) await save(draft.mount, draft.zoom);
  }
  function toggleZoom() { pick("zoom", !(draft.zoom === true)); }

  function stateLine() {
    const mc = t.model_check;
    if (t.state === "confirmed" && mc && mc.refused) {
      return h("div", { class: "ct-state warn" }, h("span", { class: "ct-icon", "aria-hidden": "true" }, "!"),
        h("strong", null, "The cameras stage could not use this type: "), mc.why || "the data contradicts it",
        h("span", { class: "muted" }, ` — it kept ${mc.model === "per_frame" ? "the general per-frame model" : mc.model === "fixed_focal" ? "per-frame poses with one focal" : mc.model || "a safer model"}${mc.at ? " (" + fmtTime(mc.at) + ")" : ""}. Change the answer, or keep it (nothing is blocked).`));
    }
    if (t.state === "confirmed") {
      return h("div", { class: "ct-state good" }, h("span", { class: "ct-icon", "aria-hidden": "true" }, "✓"),
        "Confirmed by you", t.set_at ? h("span", { class: "muted" }, " · " + fmtTime(t.set_at)) : null,
        sugg && sugg.mount ? (classOf(sugg.mount, sugg.zoom) === classOf(t.mount, t.zoom)
          ? h("span", { class: "muted" }, " · the data agrees")
          : h("span", { class: "muted" }, ` · the probe suggested ${CLASS_LABEL[classOf(sugg.mount, sugg.zoom)] || "otherwise"}; yours is kept`)) : null);
    }
    if (t.state === "suggested" && sugg) {
      const differs = complete() && classOf(draft.mount, draft.zoom) !== classOf(sugg.mount, sugg.zoom);
      return h("div", { class: "ct-state warn" }, h("span", { class: "ct-icon", "aria-hidden": "true" }, "~"),
        `Suggested: ${CLASS_LABEL[classOf(sugg.mount, sugg.zoom)] || "?"}`,
        sugg.confidence !== null && sugg.confidence !== undefined
          ? h("span", { class: ["conf", sugg.confidence >= 0.7 ? "hi" : "lo"] }, ` ${Math.round(sugg.confidence * 100)} %`) : null,
        h("span", { class: "muted" }, " — please confirm"),
        (onAccept || differs) && complete() ? h("button", { class: "btn small primary", disabled, onclick: () => confirm(),
          title: "Save this answer (A or Enter)" }, differs ? `Confirm ${CLASS_LABEL[classOf(draft.mount, draft.zoom)]}` : "Confirm") : null);
    }
    if (t.state === "conflict" && t.conflict) {
      const c = t.conflict;
      // "use the data": the full measured suggestion when there is one, else just the conflicting field
      const other = sugg && sugg.mount && sugg.zoom !== null && sugg.zoom !== undefined ? { mount: sugg.mount, zoom: sugg.zoom }
        : c.field === "mount" ? { mount: c.measured, zoom: t.zoom } : { mount: t.mount, zoom: c.measured };
      return h("div", { class: "ct-state crit" },
        h("div", null, h("span", { class: "ct-icon", "aria-hidden": "true" }, "!"), h("strong", null, "Conflict: "),
          c.message || `you said ${c.field} = ${c.user}; the data measures ${c.measured}`),
        h("div", { class: "ct-conflict-actions" },
          h("button", { class: "btn small", disabled, onclick: () => save(t.mount, t.zoom) },
            `Keep ${CLASS_LABEL[classOf(t.mount, t.zoom)] || "mine"}`),
          h("button", { class: "btn small", disabled, onclick: () => save(other.mount, other.zoom) },
            `Use ${CLASS_LABEL[classOf(other.mount, other.zoom)] || "measured"}`)),
        h("div", { class: "muted small" }, "Camera stages for this video stay blocked until you choose."));
    }
    const missing = !draft.mount ? "mount" : (draft.zoom === null || draft.zoom === undefined) ? "zoom" : null;
    return h("div", { class: "ct-state muted" }, h("span", { class: "ct-icon", "aria-hidden": "true" }, "?"),
      missing && (draft.mount || draft.zoom !== null && draft.zoom !== undefined)
        ? `Choose ${missing} to save`
        : "Not set: the pipeline uses the general model (handheld + zoom) until you choose");
  }

  function effectLine() {
    const eff = (t.effective && t.effective.class) || "handheld_zoom";
    const cls = classOf(draft.mount, draft.zoom) || eff;
    const text = (classes && classes[cls]) || CLASS_EFFECT[cls];
    const fx = t.effects && eff === cls && t.effects.length ? t.effects : null;
    const d = eff === cls ? t.estimate_delta : null;
    const pending = cls !== eff;
    return h("div", { class: "ct-effect" },
      h("span", { class: "muted" }, pending ? "If confirmed: " : "Pipeline: "), h("strong", null, CLASS_LABEL[cls] || cls), " — ",
      fx ? fx.join("; ") : text,
      pending ? h("span", { class: "muted" }, ` (until then: ${CLASS_LABEL[eff] || eff})`) : null,
      d && d.pod_min_p50 ? h("span", { class: ["delta", d.pod_min_p50 < 0 ? "good" : "warn"], title: d.basis || "" },
        ` ${d.pod_min_p50 > 0 ? "+" : ""}${Math.round(d.pod_min_p50)} pod-min (${d.usd_p50 > 0 ? "+" : ""}${fmtUSD(d.usd_p50)}) est.`) : null);
  }

  function render() {
    el.replaceChildren(
      h("div", { class: "ct-row" },
        h("div", { class: "ct-field" }, h("span", { class: "ct-label" }, "Mount"),
          seg("Mount", [["handheld", "Handheld", "H"], ["tripod", "Tripod", "T"]], draft.mount,
            { provisional, disabled, onPick: (v) => pick("mount", v) })),
        h("div", { class: "ct-field", title: "Did the focal length change during this video (pinch / zoom buttons)? Every phone CAN zoom; this asks whether it did." },
          h("span", { class: "ct-label" }, "Zoomed during the video"),
          seg("Zoomed during the video", [[false, "No", "Z"], [true, "Yes", "Z"]], draft.zoom,
            { provisional, disabled, onPick: (v) => pick("zoom", v) })),
        t.pans ? h("span", { class: "pill pill-info", title: "automatic: the tripod head pans" }, "pans") : null),
      stateLine(),
      compact ? null : h("div", { class: "ct-sugg" }, sugg
        ? [h("span", { class: "muted" }, `Suggestion (${BY[sugg.by] || sugg.by || "auto"}): `),
          CLASS_LABEL[classOf(sugg.mount, sugg.zoom)] || "?",
          sugg.confidence !== null && sugg.confidence !== undefined ? ` · ${Math.round(sugg.confidence * 100)} %` : "",
          evidenceList(sugg)]
        : h("span", { class: "muted" }, "No automatic suggestion yet (Re-probe to compute one).")),
      compact ? null : effectLine());
  }
  render();
  return { el, pick, confirm, toggleZoom, get draft() { return draft; } };
}
