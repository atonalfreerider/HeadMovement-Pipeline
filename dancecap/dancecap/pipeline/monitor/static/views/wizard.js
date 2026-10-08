// New-capture wizard (#/new, PIPELINE_MONITOR.md 6.3): 1 Sources (videos + soundtrack + take name + capture type),
// 2 Videos & camera types (probe suggestions pre-filled, you confirm), 3 Dancers (statures, shoes, re-ID description,
// portrait), 4 Dance (level, category, song), 5 Consent, 6 Review (TOML preview from a dry run) and Create.
// Files are picked through the native dialog on the server (browsers never reveal real paths).
import { h, replace, fmtBytes, fmtClock, session, debounce } from "../util.js";
import { get, post, seg } from "../api.js";
import { pill } from "../components/status.js";
import { toast, errorText, confirmDialog } from "../components/dialog.js";
import { cameraTypeControl, CLASS_LABEL, classOf } from "../components/cameraType.js";

const STEPS = ["Sources", "Videos & camera types", "Dancers", "Dance", "Consent", "Review & create"];
const LEVELS = ["professional", "intermediate", "novice"];
const CATEGORIES = [["demo", "demo"], ["lesson", "lesson"], ["jack_and_jill", "jack & jill"]];
const SCOPES = [["capture", "capture"], ["fit_3d", "3D fit"], ["ml_analysis", "ML analysis"], ["cloud_processing", "cloud (Runpod)"],
  ["name_shown", "name shown"], ["level_shown", "level shown"], ["publication", "publication"], ["public_repo", "public repo"]];
const SHOES = [["barefoot", "barefoot (0 cm)", 0], ["sneakers", "sneakers (2.5 cm)", 0.025], ["heels", "dance heels (6 cm)", 0.06], ["custom", "custom…", null]];
const TAKE_RE = /^[a-z0-9][a-z0-9_]{0,47}$/;

function blank() {
  const dancer = () => ({ name: "", stature: "", shoe: "sneakers", shoe_cm: "", description: "", portrait: null });
  const consent = () => Object.fromEntries(SCOPES.map(([k]) => [k, k === "public_repo" ? false : "?"]));
  return { step: 0, videos: [], soundtrack: null, take: "", takeEdited: false, source_mode: "reference", capture_type: "auto",
    folder: "", dancers: { lead: dancer(), follow: dancer() },
    dance: { level: "professional", category: "demo", song: "", artist: "", date: "", venue: "" },
    consent: { lead: consent(), follow: consent(), source_rights: "", bystanders: "" }, preview: null, problems: [] };
}

/** Same rules as intake.parse_stature: 5'8" | 5 ft 8 in | 173 cm | 173 | 1.727 | 1.727 m -> metres. */
export function parseStature(v) {
  if (v === null || v === undefined || v === "") return null;
  const s = String(v).trim().toLowerCase().replace(/[’′]/g, "'").replace(/[”″]/g, '"');
  let m = s.match(/^(\d+)\s*(?:'|ft|feet)\s*(?:(\d+(?:\.\d+)?)\s*(?:"|in|inch|inches|'')?)?$/);
  if (m) return Math.round((parseInt(m[1], 10) * 0.3048 + parseFloat(m[2] || "0") * 0.0254) * 1000) / 1000;
  m = s.match(/^(\d+(?:\.\d+)?)\s*(cm|m)?$/);
  if (m) {
    const x = parseFloat(m[1]);
    const unit = m[2] || (x > 3 ? "cm" : "m");
    return Math.round((unit === "cm" ? x / 100 : x) * 1000) / 1000;
  }
  return null;
}

function slug(s) {
  return String(s || "").toLowerCase().normalize("NFKD").replace(/[\u0300-\u036f]/g, "").replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "").slice(0, 48);
}

function camFromName(name, used) {
  const m = String(name).match(/^(\d{2})(?!\d)/);
  if (m && !used.has(m[1])) return m[1];
  for (let i = 1; i < 100; i++) { const c = String(i).padStart(2, "0"); if (!used.has(c)) return c; }
  return "99";
}

export default {
  title: () => "New capture",
  async mount(root, route, app) {
    let w = session.get("wizard", null) || blank();
    const save = debounce(() => session.set("wizard", { ...w, preview: null }), 200);
    const stepper = h("div", { class: "stepper", role: "tablist" });
    const body = h("div", { class: "wiz-body" });
    const foot = h("div", { class: "wiz-foot" });
    replace(root,
      h("div", { class: "page-head" }, h("h1", null, "New capture"),
        h("span", { class: "muted" }, "videos (1, 2 or many cameras) + the soundtrack → a take ready for the pipeline"),
        h("button", { class: "btn ghost", onclick: async () => {
          if (await confirmDialog("Start over?", "Clears this form (picked files stay on disk).", { okLabel: "Clear" })) { w = blank(); session.del("wizard"); draw(); }
        } }, "Start over")),
      stepper, body, foot);
    let probeTimer = null;

    const usedVideos = () => w.videos.filter((v) => v.use);
    const variantOf = () => {
      const n = usedVideos().length;
      if (w.capture_type !== "auto") return w.capture_type;
      return n >= 3 ? "multi" : n === 2 ? "two" : n === 1 ? "single" : "?";
    };

    function stepOk(i) {
      if (i === 0) return w.videos.length > 0 && TAKE_RE.test(w.take) && !takeExists();
      if (i === 1) return usedVideos().length > 0 && camsUnique();
      if (i === 2) return ["lead", "follow"].every((r) => w.dancers[r].name.trim() && statureOk(w.dancers[r].stature));
      if (i === 3) return !!w.dance.level && !!w.dance.category;
      return true;
    }
    const takeExists = () => ((app.state && app.state.takes) || []).some((t) => t.take === w.take);
    const camsUnique = () => { const c = usedVideos().map((v) => v.cam); return new Set(c).size === c.length && c.every((x) => /^\d{2}$/.test(x)); };
    const statureOk = (s) => { const v = parseStature(s); return v !== null && v >= 1.2 && v <= 2.2; };

    function draw() {
      save();
      replace(stepper, STEPS.map((label, i) => h("button", { role: "tab", "aria-selected": String(i === w.step),
        class: [i === w.step ? "active" : "", i !== w.step && i < w.step ? (stepOk(i) ? "ok" : "bad") : ""],
        onclick: () => go(i) }, h("span", { class: "n" }, i < w.step && stepOk(i) ? "✓" : String(i + 1)), label)));
      replace(body, [stepSources, stepVideos, stepDancers, stepDance, stepConsent, stepReview][w.step]());
      replace(foot,
        w.step > 0 ? h("button", { class: "btn", onclick: () => go(w.step - 1) }, "← Back") : null,
        h("span", { class: "spacer" }),
        !stepOk(w.step) && w.step < 5 ? h("span", { class: "muted small" }, stepHint(w.step)) : null,
        w.step < 5 ? h("button", { class: "btn primary", onclick: () => go(w.step + 1), disabled: !stepOk(w.step) }, `Next: ${STEPS[w.step + 1]} →`) : null);
      schedulePoll();
    }
    function stepHint(i) {
      if (i === 0) return !w.videos.length ? "pick at least one video" : !TAKE_RE.test(w.take) ? "take name: lowercase letters, digits, _" : takeExists() ? "that take name exists" : "";
      if (i === 1) return !usedVideos().length ? "use at least one video" : "camera ids must be 2 unique digits";
      if (i === 2) return "a name and a plausible stature for both dancers";
      return "";
    }
    function go(i) {
      if (i > w.step) for (let k = w.step; k < i; k++) if (!stepOk(k)) { w.step = k; draw(); return; }
      w.step = Math.max(0, Math.min(5, i));
      draw();
      root.closest(".main") && (root.closest(".main").scrollTop = 0);
    }

    // ---------------------------------------------------------------------------------------------- files
    function addFiles(files, folder) {
      const usedCams = new Set(w.videos.map((v) => v.cam));
      for (const f of files || []) {
        if (f.kind === "video") {
          if (w.videos.some((v) => v.handle === f.handle)) continue;
          const cam = camFromName(f.name, usedCams);
          usedCams.add(cam);
          w.videos.push({ ...f, cam, use: true, type: null });
        } else if (f.kind === "audio" && !w.soundtrack) w.soundtrack = f;
      }
      w.videos.sort((a, b) => a.cam.localeCompare(b.cam));
      if (folder && !w.takeEdited) w.take = slug(String(folder).split(/[\\/]/).filter(Boolean).pop());
      draw();
    }
    async function pick(kind) {
      try {
        toast("A file dialog opened on your desktop (it may be behind this window).", "info", { timeout: 4000 });
        const r = await post("/api/fs/pick", { kind });
        if (kind === "folder") { w.folder = r.folder || w.folder; addFiles(r.files, r.folder); return; }
        if (kind === "soundtrack") { if (r.files && r.files[0]) { w.soundtrack = r.files[0]; draw(); } return; }
        addFiles(r.files);
      } catch (e) { toast(errorText(e), "crit"); }
    }
    async function scan() {
      try { const r = await post("/api/fs/scan", { folder: w.folder }); addFiles(r.files, r.folder); }
      catch (e) { toast(errorText(e), "crit"); }
    }
    function schedulePoll() {
      clearTimeout(probeTimer);
      const pending = w.videos.filter((v) => v.probe_state === "queued" || v.probe_state === "running");
      if (w.soundtrack && (w.soundtrack.probe_state === "queued" || w.soundtrack.probe_state === "running")) pending.push(w.soundtrack);
      if (!pending.length) return;
      probeTimer = setTimeout(async () => {
        let changed = false;
        for (const p of pending) {
          try {
            const f = await get(`/api/staging/${seg(p.handle)}`);
            if (f.probe_state !== p.probe_state || JSON.stringify(f.suggestion) !== JSON.stringify(p.suggestion)) { Object.assign(p, f); changed = true; }
          } catch { /* handle expired: leave it */ }
        }
        if (changed && (w.step <= 1 || w.step === 5)) draw(); else schedulePoll();
      }, 2000);
    }

    // ---------------------------------------------------------------------------------------------- step 1
    function stepSources() {
      const folderIn = h("input", { type: "text", value: w.folder, placeholder: "C:\\...\\dances\\<capture folder>", style: { flex: 1, minWidth: "320px" } });
      folderIn.addEventListener("input", () => { w.folder = folderIn.value; save(); });
      folderIn.addEventListener("keydown", (e) => { if (e.key === "Enter") scan(); });
      const takeIn = h("input", { type: "text", value: w.take, placeholder: "e.g. larissa_kadu_2026_11", class: ["mono", w.take && (!TAKE_RE.test(w.take) || takeExists()) ? "invalid" : ""] });
      takeIn.addEventListener("input", () => { w.take = takeIn.value.trim(); w.takeEdited = true; save(); refreshFoot(); takeIn.classList.toggle("invalid", !!w.take && (!TAKE_RE.test(w.take) || takeExists())); });
      const total = w.videos.reduce((a, v) => a + (v.size_bytes || 0), 0);
      const n = usedVideos().length;
      return [
        h("div", { class: "card" },
          h("h3", { class: "section-title" }, "Videos"),
          h("div", { class: "row" },
            h("button", { class: "btn primary", onclick: () => pick("videos") }, "Browse videos…"),
            h("button", { class: "btn", onclick: () => pick("folder") }, "Pick a folder…"),
            h("span", { class: "muted" }, "or paste a folder:"), folderIn, h("button", { class: "btn", onclick: scan }, "Scan")),
          w.videos.length ? h("div", { class: "file-list", style: { marginTop: "10px" } },
            w.videos.map((v) => h("div", { class: "file-row" }, h("span", { class: "mono" }, v.cam), h("span", { class: "ellipsis", style: { flex: 1 }, title: v.path }, v.name),
              h("span", { class: "muted small" }, fmtBytes(v.size_bytes)), probeChip(v),
              h("button", { class: "icon-btn", "aria-label": "Remove", title: "Remove from this capture", onclick: () => { w.videos = w.videos.filter((x) => x !== v); draw(); } }, "✕"))),
            h("div", { class: "small muted" }, `${w.videos.length} file${w.videos.length > 1 ? "s" : ""}, ${fmtBytes(total)}`))
            : h("p", { class: "muted" }, "One video gives a single-view capture, two a two-view capture, three or more a multi-view capture.")),
        h("div", { class: "card" },
          h("h3", { class: "section-title" }, "Soundtrack"),
          h("div", { class: "row" },
            h("button", { class: "btn", onclick: () => pick("soundtrack") }, "Browse audio…"),
            w.soundtrack ? [h("strong", null, w.soundtrack.name), w.soundtrack.probe && w.soundtrack.probe.duration_s ? h("span", { class: "muted" }, fmtClock(w.soundtrack.probe.duration_s).replace(/\.\d$/, "")) : null,
              h("button", { class: "btn small ghost", onclick: () => { w.soundtrack = null; draw(); } }, "Remove")]
              : h("span", { class: "muted" }, "optional — with it, the phone clocks align to the studio track (±1-2 ms) and the beat grid comes from it")),
          h("p", { class: "small muted" }, "The soundtrack stays on this PC: never in a repository or a public build.")),
        h("div", { class: "card" },
          h("h3", { class: "section-title" }, "Take"),
          h("div", { class: "form-grid" },
            h("label", { class: "field" }, h("span", null, "Take name (slug, unique)"), takeIn),
            h("div", { class: "field" }, h("span", null, "Sources"), h("div", { class: "radio-chips" },
              radio("srcmode", "reference", w.source_mode, "use in place", (v) => { w.source_mode = v; save(); }),
              radio("srcmode", "copy", w.source_mode, "copy to dances/<take>/", (v) => { w.source_mode = v; save(); }))),
            h("div", { class: "field" }, h("span", null, "Capture type"), h("div", { class: "radio-chips" },
              [["auto", `auto (${n >= 3 ? "multi" : n === 2 ? "two" : n === 1 ? "single" : "?"})`], ["multi", "multi-view"], ["two", "two-view"], ["single", "single-view"]].map(([v, l]) =>
                radio("ctype", v, w.capture_type, l, (x) => { w.capture_type = x; draw(); }, v !== "auto" && !variantAllowed(v)))))),
          variantNote()),
      ];
    }
    function variantAllowed(v) {
      const n = usedVideos().length;
      const need = { multi: 3, two: 2, single: 1 }[v];
      return n >= need;
    }
    function variantNote() {
      const v = variantOf();
      const text = {
        multi: "Multi-view: cameras from the static room first (SfM + per-frame localisation), video sync, then the chunked multi-view body fit (MAMMA) on Runpod.",
        two: "Two-view: the pair is calibrated from the dancers' keypoints (epipolar sync, scale from statures), then a 2-camera fit with up to 2 refinement passes.",
        single: "Single-view: the camera model follows its type (tripod = no visual odometry), monocular body fits for the couple, then a joint refinement with the known statures (fixes the monocular scale).",
      }[v];
      return text ? h("p", { class: "small", style: { marginTop: "10px", color: "var(--text-2)" } }, text) : null;
    }
    function refreshFoot() {
      replace(foot, h("span", { class: "spacer" }), !stepOk(w.step) ? h("span", { class: "muted small" }, stepHint(w.step)) : null,
        h("button", { class: "btn primary", onclick: () => go(w.step + 1), disabled: !stepOk(w.step) }, `Next: ${STEPS[w.step + 1]} →`));
    }

    function probeChip(v) {
      if (v.probe_state === "done") return pill("probed", "good");
      if (v.probe_state === "failed") return pill("probe failed", "crit", v.error || "");
      return pill(v.probe_state === "running" ? "probing…" : "queued", "info");
    }

    // ---------------------------------------------------------------------------------------------- step 2
    function wizType(v) {
      const sg = v.suggestion || null;
      if (v.type) return { cam: v.cam, mount: v.type.mount, zoom: v.type.zoom, source: "user", state: "confirmed", suggestion: sg,
        effective: { mount: v.type.mount, zoom: v.type.zoom, class: classOf(v.type.mount, v.type.zoom) } };
      if (sg) return { cam: v.cam, mount: null, zoom: null, source: "auto", state: "suggested", suggestion: sg,
        effective: { mount: sg.mount, zoom: sg.zoom, class: classOf(sg.mount, sg.zoom) } };
      return { cam: v.cam, mount: null, zoom: null, source: "default", state: "unset", suggestion: null,
        effective: { mount: "handheld", zoom: true, class: "handheld_zoom" } };
    }
    function stepVideos() {
      if (!w.videos.length) return h("div", { class: "empty" }, "No videos picked yet: go back to Sources.");
      const acceptable = w.videos.filter((v) => v.use && !v.type && v.suggestion && (v.suggestion.confidence ?? 0) >= 0.7);
      const dup = new Set();
      const seen = new Set();
      for (const v of usedVideos()) { if (seen.has(v.cam)) dup.add(v.cam); seen.add(v.cam); }
      return [
        h("div", { class: "row" },
          h("span", null, `${usedVideos().length} of ${w.videos.length} videos used → `, h("strong", null, variantOf()), " capture"),
          h("span", { class: "spacer" }),
          h("button", { class: "btn", disabled: !acceptable.length, onclick: () => { for (const v of acceptable) v.type = { mount: v.suggestion.mount, zoom: v.suggestion.zoom }; draw(); } },
            `Accept suggestions ≥ 70 % (${acceptable.length})`)),
        h("p", { class: "small muted" }, "For each video: was the phone handheld or on a tripod, and did it zoom? The probe pre-fills a suggestion from the background motion; confirm it or correct it. Unset videos use the general model (handheld + zoom)."),
        h("div", { class: "vgrid" }, w.videos.map((v) => {
          const pr = v.probe || {};
          const camIn = h("input", { type: "text", value: v.cam, maxlength: "2", class: ["cam-input", dup.has(v.cam) || !/^\d{2}$/.test(v.cam) ? "invalid" : ""], "aria-label": "camera id" });
          camIn.addEventListener("change", () => { v.cam = camIn.value.trim().padStart(2, "0"); draw(); });
          const useBox = h("input", { type: "checkbox", checked: v.use });
          useBox.addEventListener("change", () => { v.use = useBox.checked; draw(); });
          const ctl = cameraTypeControl({ type: wizType(v), disabled: !v.use,
            onSave: async (m, z) => { v.type = { mount: m, zoom: z }; draw(); },
            onAccept: async () => { v.type = { mount: v.suggestion.mount, zoom: v.suggestion.zoom }; draw(); } });
          return h("div", { class: ["vcard", v.use ? "" : "excluded", v.type ? "" : v.suggestion ? "state-suggested" : ""] },
            h("div", { class: "vc-head" }, h("label", { class: "chk", title: "use this video" }, useBox), h("span", { class: "muted small" }, "cam"), camIn,
              h("div", { class: "vc-file ellipsis", title: v.path }, h("strong", null, v.name)), probeChip(v)),
            h("div", { class: "vc-meta" },
              pr.display_w ? h("span", null, `${pr.display_w}×${pr.display_h}`) : null,
              pr.fps ? h("span", null, `${pr.fps.toFixed(2)} fps${pr.vfr ? " VFR" : ""}`) : null,
              pr.duration_s ? h("span", null, fmtClock(pr.duration_s).replace(/\.\d$/, "")) : null,
              pr.codec ? h("span", { class: pr.playable === false ? "warn-text" : "" }, pr.codec + (pr.playable === false ? " (not playable here: proxy later)" : "")) : null,
              pr.audio !== undefined ? h("span", null, pr.audio ? "audio ✓" : h("span", { class: "warn-text" }, "no audio (no clock sync from sound)")) : null,
              h("span", null, fmtBytes(v.size_bytes))),
            v.strip ? h("div", { class: "vc-media" }, h("img", { class: "strip-img", src: v.strip, alt: "", loading: "lazy", onerror: (e) => e.target.remove() })) : null,
            ctl.el);
        })),
        dup.size ? h("p", { class: "crit-text" }, `Camera ids must be unique: ${Array.from(dup).join(", ")}`) : null,
      ];
    }

    // ---------------------------------------------------------------------------------------------- step 3
    function stepDancers() {
      return h("div", { class: "grid2" }, ["lead", "follow"].map((role) => {
        const d = w.dancers[role];
        const name = h("input", { type: "text", value: d.name, placeholder: role === "lead" ? "e.g. Kadu" : "e.g. Larissa" });
        name.addEventListener("input", () => { d.name = name.value; save(); refreshFoot(); });
        const st = h("input", { type: "text", value: d.stature, placeholder: `5'8"  or  173 cm  or  1.73`, class: d.stature && !statureOk(d.stature) ? "invalid" : "" });
        const stOut = h("span", { class: "small" });
        const upd = () => {
          const v = parseStature(st.value);
          st.classList.toggle("invalid", !!st.value && !statureOk(st.value));
          stOut.textContent = v === null ? (st.value ? "cannot read this" : "") : `= ${v.toFixed(3)} m${v < 1.2 || v > 2.2 ? " (implausible)" : ""}`;
          stOut.className = "small " + (v !== null && v >= 1.2 && v <= 2.2 ? "good-text" : "crit-text");
        };
        st.addEventListener("input", () => { d.stature = st.value; upd(); save(); refreshFoot(); });
        upd();
        const shoe = h("select", null, SHOES.map(([k, l]) => h("option", { value: k, selected: d.shoe === k }, l)));
        const shoeCm = h("input", { type: "number", min: "0", max: "10", step: "0.5", value: d.shoe_cm, placeholder: "cm", style: { width: "80px" }, hidden: d.shoe !== "custom" });
        shoe.addEventListener("change", () => { d.shoe = shoe.value; shoeCm.hidden = d.shoe !== "custom"; save(); });
        shoeCm.addEventListener("input", () => { d.shoe_cm = shoeCm.value; save(); });
        const desc = h("textarea", { rows: "2", placeholder: "clothes, hair… (tells the two dancers apart for re-identification)" }, d.description);
        desc.addEventListener("input", () => { d.description = desc.value; save(); });
        return h("div", { class: "card" },
          h("div", { class: "card-head" }, h("h3", null, role === "lead" ? "Lead" : "Follow")),
          h("div", { class: "form-grid" },
            h("label", { class: "field" }, h("span", null, "Name"), name),
            h("label", { class: "field" }, h("span", null, "Stature"), st, stOut),
            h("label", { class: "field" }, h("span", null, "Shoes"), h("div", { class: "row" }, shoe, shoeCm))),
          h("label", { class: "field", style: { marginTop: "10px" } }, h("span", null, "Description (re-ID)"), desc),
          h("div", { class: "row", style: { marginTop: "10px" } },
            h("span", { class: "flabel" }, "Portrait"),
            d.portrait ? [h("img", { src: `/api/media/${seg(d.portrait.handle)}/thumb?w=160`, alt: "", style: { height: "56px", borderRadius: "6px" } }),
              h("span", { class: "small" }, d.portrait.name), h("button", { class: "btn small ghost", onclick: () => { d.portrait = null; draw(); } }, "Remove")]
              : h("span", { class: "muted small" }, "optional: a clear face photo improves the avatar's face"),
            h("button", { class: "btn small", onclick: async () => {
              try { const r = await post("/api/fs/pick", { kind: "portrait" }); if (r.files && r.files[0]) { d.portrait = r.files[0]; draw(); } }
              catch (e) { toast(errorText(e), "crit"); }
            } }, "Browse…")),
          h("p", { class: "small muted" }, "The stature fixes the metric scale of the 3D fit (and the monocular scale for single-view captures)."));
      }));
    }

    // ---------------------------------------------------------------------------------------------- step 4
    function stepDance() {
      const d = w.dance;
      const inp = (k, ph, type = "text") => {
        const el = h("input", { type, value: d[k] || "", placeholder: ph });
        el.addEventListener("input", () => { d[k] = el.value; save(); titleOut.textContent = titlePreview(); });
        return el;
      };
      const titleOut = h("strong", null, titlePreview());
      return h("div", { class: "card" },
        h("div", { class: "form-grid" },
          h("div", { class: "field" }, h("span", null, "Level"), h("div", { class: "radio-chips" }, LEVELS.map((l) => radio("level", l, d.level, l, (v) => { d.level = v; save(); })))),
          h("div", { class: "field" }, h("span", null, "Category"), h("div", { class: "radio-chips" }, CATEGORIES.map(([v, l]) => radio("cat", v, d.category, l, (x) => { d.category = x; save(); }))))),
        h("div", { class: "form-grid", style: { marginTop: "12px" } },
          h("label", { class: "field" }, h("span", null, "Song"), inp("song", "song title")),
          h("label", { class: "field" }, h("span", null, "Artist"), inp("artist", "artist")),
          h("label", { class: "field" }, h("span", null, "Date"), inp("date", "", "date")),
          h("label", { class: "field" }, h("span", null, "Venue"), inp("venue", "optional"))),
        h("p", { style: { marginTop: "12px" } }, h("span", { class: "muted" }, "Title: "), titleOut));
    }
    function titlePreview() {
      const l = w.dancers.lead.name.trim() || "Lead", f = w.dancers.follow.name.trim() || "Follow";
      return `${l} + ${f}${w.dance.song.trim() ? " x " + w.dance.song.trim() : ""}`;
    }

    // ---------------------------------------------------------------------------------------------- step 5
    function stepConsent() {
      const c = w.consent;
      const label = (v) => (v === true ? "yes" : v === false ? "no" : "?");
      const tri = (role, k) => {
        const b = h("button", { type: "button", class: ["tri", c[role][k] === true ? "yes" : c[role][k] === false ? "no" : ""] }, label(c[role][k]));
        b.addEventListener("click", () => { const v = c[role][k]; c[role][k] = v === true ? false : v === false ? "?" : true; draw(); });
        return b;
      };
      const rights = h("input", { type: "text", value: c.source_rights, placeholder: "who owns the footage / permission" });
      rights.addEventListener("input", () => { c.source_rights = rights.value; save(); });
      const bys = h("input", { type: "text", value: c.bystanders, placeholder: "bystanders visible? blurred?" });
      bys.addEventListener("input", () => { c.bystanders = bys.value; save(); });
      const cloudOk = c.lead.cloud_processing === true && c.follow.cloud_processing === true;
      return h("div", { class: "card" },
        h("p", { class: "muted" }, "Recorded in the take TOML [consent]. Click to cycle yes → no → ?. \"?\" blocks the stages that need that scope."),
        h("table", { class: "tbl compact consent-tbl" },
          h("thead", null, h("tr", null, h("th", null, "dancer"), SCOPES.map(([, l]) => h("th", null, l)))),
          h("tbody", null, ["lead", "follow"].map((r) => h("tr", null,
            h("td", null, h("strong", null, r), " ", h("span", { class: "muted" }, w.dancers[r].name)),
            SCOPES.map(([k]) => h("td", null, tri(r, k))))))),
        h("p", { class: cloudOk ? "good-text" : "warn-text" }, cloudOk ? "✓ cloud_processing: every pod stage may run on Runpod."
          : "! cloud_processing is required for any Runpod stage (all heavy CV / neural work): without it the pipeline stops at the first pod stage."),
        h("div", { class: "form-grid" },
          h("label", { class: "field" }, h("span", null, "Source rights"), rights),
          h("label", { class: "field" }, h("span", null, "Bystanders"), bys)));
    }

    // ---------------------------------------------------------------------------------------------- step 6
    function request(dry) {
      const dancers = {};
      for (const r of ["lead", "follow"]) {
        const d = w.dancers[r];
        const shoe = SHOES.find(([k]) => k === d.shoe);
        const st = parseStature(d.stature);
        dancers[r] = { name: d.name.trim(), stature_m: st !== null ? st : d.stature, shoe_sole_m: shoe && shoe[2] !== null ? shoe[2] : (parseFloat(d.shoe_cm) || 0) / 100,
          description: d.description.trim(), portrait: d.portrait ? { handle: d.portrait.handle } : null };
      }
      const consent = { lead: w.consent.lead, follow: w.consent.follow, source_rights: w.consent.source_rights.trim() || "?", bystanders: w.consent.bystanders.trim() || "?" };
      return { dry_run: dry, take: w.take, source_mode: w.source_mode, capture_type: w.capture_type,
        videos: w.videos.map((v) => ({ handle: v.handle, cam: v.cam, excluded: !v.use, type: v.type ? { mount: v.type.mount, zoom: v.type.zoom } : undefined })),
        soundtrack: w.soundtrack ? { handle: w.soundtrack.handle } : null, dancers, dance: { ...w.dance }, consent };
    }
    let previewSeq = 0;
    async function runPreview(out) {
      const seq = ++previewSeq;
      replace(out, h("div", { class: "loading" }, "checking (dry run, nothing is written)…"));
      try {
        const r = await post("/api/takes", request(true));
        if (seq !== previewSeq) return;
        w.preview = r; w.problems = r.problems || [];
      } catch (e) {
        if (seq !== previewSeq) return;
        w.preview = null;
        w.problems = (e.details && e.details.problems) || [{ field: "", message: errorText(e) }];
      }
      replace(out, reviewBody());
    }
    function checklist() {
      const used = usedVideos();
      const ct = { set: used.filter((v) => v.type).length, sugg: used.filter((v) => !v.type && v.suggestion).length };
      const cloudOk = w.consent.lead.cloud_processing === true && w.consent.follow.cloud_processing === true;
      const items = [
        [used.length > 0 && camsUnique(), `${used.length} video${used.length === 1 ? "" : "s"} used (${variantOf()} capture), camera ids unique`],
        [TAKE_RE.test(w.take) && !takeExists(), `take name "${w.take}"`],
        [true, w.soundtrack ? `soundtrack: ${w.soundtrack.name}` : "no soundtrack (clocks from the phones' own audio)", !w.soundtrack],
        [["lead", "follow"].every((r) => statureOk(w.dancers[r].stature)), `statures: ${["lead", "follow"].map((r) => { const v = parseStature(w.dancers[r].stature); return `${r} ${v ? v.toFixed(3) + " m" : "?"}`; }).join(", ")}`],
        [true, `camera types: ${ct.set} set by you, ${ct.sugg} suggested (confirm later in Video Review), ${used.length - ct.set - ct.sugg} unset`, ct.set < used.length],
        [true, cloudOk ? "cloud_processing consent: yes for both dancers" : "cloud_processing consent missing: Runpod stages will wait", !cloudOk],
      ];
      return h("ul", { class: "checklist" }, items.map(([ok, text, warn]) => h("li", null,
        h("span", { class: ok && !warn ? "good-text" : ok ? "warn-text" : "crit-text", "aria-hidden": "true" }, ok && !warn ? "✓" : ok ? "!" : "✕"), text)));
    }
    function reviewBody() {
      const probs = w.problems || [];
      return h("div", { class: "grid2" },
        h("div", { class: "card" }, h("h3", { class: "section-title" }, "Checklist"), checklist(),
          probs.length ? h("div", { style: { marginTop: "10px" } }, h("strong", { class: "crit-text" }, "Problems"),
            h("ul", { class: "small crit-text" }, probs.map((p) => h("li", null, p.field ? h("code", null, p.field) : null, " ", p.message)))) : null,
          w.preview && (w.preview.warnings || []).length ? h("ul", { class: "small warn-text" }, w.preview.warnings.map((x) => h("li", null, typeof x === "string" ? x : x.message || JSON.stringify(x)))) : null,
          h("p", { class: "small muted", style: { marginTop: "10px" } }, "Create writes takes/<take>.toml and starts the probe (camera-type suggestions). Then confirm the camera types in Video Review and plan the pipeline.")),
        h("div", { class: "card" }, h("h3", { class: "section-title" }, `takes/${w.take || "<take>"}.toml (preview)`),
          w.preview && w.preview.toml_preview ? h("pre", { class: "toml-preview" }, w.preview.toml_preview) : h("p", { class: "muted" }, "No preview (fix the problems first).")));
    }
    function stepReview() {
      const out = h("div");
      runPreview(out);
      const createBtn = h("button", { class: "btn primary" }, "Create take");
      createBtn.addEventListener("click", async () => {
        if (!w.preview) { toast("Fix the problems first.", "warn"); return; }
        createBtn.disabled = true; createBtn.classList.add("busy");
        try {
          const r = await post("/api/takes", request(false));
          toast([`Take ${r.take} created. The probe run started: `, r.run ? h("a", { href: `#/runs/${r.run.id}` }, r.run.id) : ""], "good");
          session.del("wizard");
          w = blank();
          app.refreshNow();
          app.navigate(`#/take/${r.take}/videos`);
        } catch (e) {
          w.problems = (e.details && e.details.problems) || [{ field: "", message: errorText(e) }];
          replace(out, reviewBody());
          toast(errorText(e), "crit");
        } finally { createBtn.disabled = false; createBtn.classList.remove("busy"); }
      });
      return [h("div", { class: "row" }, h("button", { class: "btn", onclick: () => runPreview(out) }, "Check again"), h("span", { class: "spacer" }), createBtn), out];
    }

    function radio(name, value, cur, label, onPick, disabled) {
      const inp = h("input", { type: "radio", name, value, checked: cur === value, disabled: !!disabled });
      inp.addEventListener("change", () => { if (inp.checked) onPick(value); });
      return h("label", { style: disabled ? { opacity: 0.45 } : null, title: disabled ? "needs more videos" : null }, inp, label);
    }

    draw();
    return { destroy() { clearTimeout(probeTimer); } };
  },
};

