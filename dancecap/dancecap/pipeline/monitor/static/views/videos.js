// Video Review (#/take/<take>/videos, PIPELINE_MONITOR.md 6.4): one card per source video with a thumbnail strip,
// a player (at most 2 at once), probe facts, and the CAMERA TYPE input (Handheld | Tripod, Zoom No | Yes) pre-filled
// from the automatic suggestion, which the user confirms. Saving shows which done stages a change makes stale.
// Keys: J/K next/previous, H/T mount, Z zoom, A accept, Enter confirm, Space play/pause, N narrate.
import { h, replace, fmtDur, fmtUSD, fmtBytes, fmtClock, fingerprint, prefs } from "../util.js";
import { getx, get, put, post, patch, seg, listOf, forget } from "../api.js";
import { pill } from "../components/status.js";
import { toast, errorText, confirmDialog } from "../components/dialog.js";
import { cameraTypeControl, cameraTypeChip, CLASS_LABEL, CLASS_EFFECT, classOf } from "../components/cameraType.js";
import { takeHeader } from "./takehead.js";

const MAX_PLAYERS = 2;

export default {
  title: (r) => `${r.take} videos`,
  async mount(root, route, app) {
    const take = route.take;
    let detail = null, videos = [], etag = null, classes = null, loaded = false;
    let sel = 0;
    let fpHead = null, fpList = null;
    const cards = new Map(); // cam -> {el, ctl, v, fp, ctHost, headHost, metaHost, footHost, media}
    const players = []; // cams with an open <video>, oldest first
    const headEl = h("div");
    const sumEl = h("div");
    const explainEl = h("div");
    const gridEl = h("div", { class: "vgrid" });
    replace(root, headEl, sumEl, explainEl, gridEl);
    app.registry().then((r) => { classes = (r.camera_types || {}).classes || null; drawExplain(); for (const c of cards.values()) c.fp = null; drawCards(); }).catch(() => {});

    async function load(force) {
      try {
        if (force) forget(`/api/takes/${take}/videos`);
        const [d, vv] = await Promise.all([getx(`/api/takes/${seg(take)}`), getx(`/api/takes/${seg(take)}/videos`)]);
        detail = d.data;
        videos = listOf(vv.data, "videos");
        etag = (vv.data && vv.data.toml_etag) || detail.toml_etag;
        loaded = true;
      } catch (e) {
        replace(gridEl, h("div", { class: "card crit-text" }, e.status === 404 ? `No take "${take}".` : errorText(e)));
        return;
      }
      const fh = fingerprint({ detail, runs: app.state && app.state.runs_active });
      if (fh !== fpHead) { fpHead = fh; replace(headEl, takeHeader(app, take, detail, "videos")); }
      drawSummary();
      drawCards();
    }

    // ---------------------------------------------------------------------------------------------- summary
    let fpSum = null;
    function drawSummary() {
      // rebuilt only when its inputs change: a rebuild every 2 s poll would drop keyboard focus and eat clicks
      const f = fingerprint(videos.map((v) => [v.cam, v.excluded, v.type, v.clock && v.clock.reference]));
      if (f === fpSum) return;
      fpSum = f;
      const used = videos.filter((v) => !v.excluded);
      const n = { confirmed: 0, suggested: 0, unset: 0, conflict: 0 };
      for (const v of used) n[(v.type && v.type.state) || "unset"]++;
      const acceptable = used.filter((v) => v.type && v.type.state === "suggested" && v.type.suggestion && (v.type.suggestion.confidence ?? 0) >= 0.7);
      const ref = videos.find((v) => v.clock && v.clock.reference);
      let dMin = 0, dUsd = 0;
      for (const v of used) { const d = v.type && v.type.estimate_delta; if (d) { dMin += d.pod_min_p50 || 0; dUsd += d.usd_p50 || 0; } }
      const byClass = {};
      for (const v of used) { const c = v.type && v.type.effective && v.type.effective.class || "handheld_zoom"; (byClass[c] = byClass[c] || []).push(v.cam); }
      replace(sumEl, h("div", { class: "card", style: { marginBottom: "12px" } },
        h("div", { class: "vr-summary" },
          h("strong", null, `${used.length} used / ${videos.length} videos`),
          h("span", { class: "row", style: { gap: "4px" } }, "camera types:",
            n.confirmed ? pill(`${n.confirmed} confirmed`, "good") : null, n.suggested ? pill(`${n.suggested} suggested`, "warn") : null,
            n.unset ? pill(`${n.unset} unset`, "neutral") : null, n.conflict ? pill(`${n.conflict} conflict`, "crit") : null),
          ref ? h("span", { class: "muted" }, `clock reference: cam ${ref.cam}`) : null,
          h("span", { class: "spacer" }),
          h("button", { class: "btn", disabled: !acceptable.length, onclick: () => acceptMany(acceptable.map((v) => v.cam)),
            title: "Confirm every suggestion with confidence of at least 70 %" }, `Accept suggestions ≥ 70 % (${acceptable.length})`),
          h("button", { class: "btn", onclick: () => reprobe(null), title: "Re-run the camera-motion probe for every video (a short local run)" }, "Re-probe all")),
        h("div", { class: "row small", style: { marginTop: "8px", gap: "6px" } },
          used.map((v, i) => h("button", { class: "btn small ghost", title: `cam ${v.cam}: ${v.type ? v.type.state : "?"}`, onclick: () => select(videos.indexOf(v), true) },
            h("span", { class: "mono" }, v.cam), cameraTypeChip(v.type))),
          h("span", { class: "spacer" }),
          h("span", { class: "muted" }, "pipeline: ", Object.entries(byClass).map(([c, cams]) => `${CLASS_LABEL[c] || c} × ${cams.length}`).join(" · ")),
          Math.abs(dMin) >= 0.5 ? h("span", { class: dMin < 0 ? "good-text" : "warn-text", title: "cam_loc work vs the general handheld + zoom model (unit count: no tripod measured yet)" },
            ` · ${dMin < 0 ? "" : "+"}${Math.round(dMin)} pod-min (${fmtUSD(dUsd)}) est.`) : null)));
    }

    function drawExplain() {
      const open = prefs.get("ct_explain_open", true);
      const det = h("details", { class: "card vr-explain", open },
        h("summary", null, h("strong", null, "How the camera type helps the pipeline"),
          h("span", { class: "muted" }, " — tell it how each video was filmed; the automatic suggestion is pre-filled for you to confirm")),
        h("div", { class: "cls" }, ["tripod", "tripod_zoom", "handheld", "handheld_zoom"].map((c) => h("div", null,
          h("strong", null, CLASS_LABEL[c]), (classes && classes[c]) || CLASS_EFFECT[c]))),
        h("ul", { class: "small", style: { margin: "10px 0 0", paddingLeft: "18px", color: "var(--text-2)" } },
          h("li", null, "A locked tripod (no zoom, no pans) is localised on keyframes (~10 % of the work) and gets one constant pose for the whole video: no per-frame camera noise for sync and the dancers. (Tripods do not anchor the room map or the world frame yet.)"),
          h("li", null, "Zoomed during the video = no fixes one focal + lens distortion per video (fewer unknowns); yes keeps a per-frame focal with log-focal smoothing. Every phone CAN zoom: the question is whether it did in this video."),
          h("li", null, "Unset and unconfirmed cameras use the general model (handheld + zoom), which is today's behaviour, until you confirm; they are priced as handheld + zoom in the plan's p90."),
          h("li", null, "The cameras stage checks every type against the data: a model the data refutes (the camera moves or zooms, or the track would get shakier) is not used - a safer model is kept and the card says why."),
          h("li", null, "You always win: when the camera track later disagrees, the card turns red with the evidence and the camera stages wait for your choice.")));
      det.addEventListener("toggle", () => prefs.set("ct_explain_open", det.open));
      replace(explainEl, det);
    }

    // ---------------------------------------------------------------------------------------------- cards
    function drawCards() {
      const order = videos.map((v) => v.cam).join(",");
      if (order !== fpList) {
        fpList = order;
        for (const c of cards.values()) c.el.remove();
        cards.clear();
        players.length = 0;
      }
      if (!videos.length) {
        if (loaded) { cards.clear(); players.length = 0; fpList = null; replace(gridEl, h("div", { class: "empty" }, "This take lists no videos.")); }
        return;
      }
      for (const el of Array.from(gridEl.children)) if (!el.classList.contains("vcard")) el.remove();
      videos.forEach((v, i) => {
        let c = cards.get(v.cam);
        if (!c) {
          c = makeCard(v, i);
          cards.set(v.cam, c);
          gridEl.appendChild(c.el);
        }
        updateCard(c, v, i);
      });
    }

    function makeCard(v, i) {
      const c = { v, i, headHost: h("div", { class: "vc-head" }), metaHost: h("div", { class: "vc-meta" }), media: h("div", { class: "vc-media" }),
        ctHost: h("div"), footHost: h("div", { class: "vc-foot" }), fp: null, mediaFp: null };
      c.el = h("div", { class: "vcard", tabindex: "-1", "data-cam": v.cam, onclick: () => select(videos.findIndex((x) => x.cam === v.cam), false) },
        c.headHost, c.metaHost, c.media, c.ctHost, c.footHost);
      return c;
    }

    function updateCard(c, v, i) {
      c.v = v; c.i = i;
      const f = fingerprint(v);
      c.el.className = ["vcard", i === sel ? "sel" : "", v.excluded ? "excluded" : "", "state-" + (v.type ? v.type.state : "unset")].join(" ");
      if (f === c.fp) return;
      c.fp = f;
      const pr = v.probe || {};
      replace(c.headHost,
        h("span", { class: "vc-cam" }, v.cam),
        h("div", { class: "vc-file" }, h("div", { class: "ellipsis", title: v.file || "" }, h("strong", null, v.file || "(file missing)"))),
        v.clock && v.clock.reference ? pill("clock reference", "info") : null,
        v.excluded ? pill("excluded", "neutral", v.exclude_reason || "") : null,
        !v.present ? pill("file missing", "crit") : null,
        cameraTypeChip(v.type));
      replace(c.metaHost,
        pr.display_w ? h("span", null, `${pr.display_w}×${pr.display_h}`) : null,
        pr.fps ? h("span", { title: pr.vfr ? "variable frame rate: never assumed constant" : "" }, `${pr.fps.toFixed(2)} fps${pr.vfr ? " VFR" : ""}`) : null,
        pr.duration_s ? h("span", null, fmtClock(pr.duration_s).replace(/\.\d$/, "")) : null,
        pr.codec ? h("span", null, pr.codec + (pr.audio ? ` + ${pr.audio_codec || "audio"}` : ", no audio")) : null,
        pr.size_bytes ? h("span", null, fmtBytes(pr.size_bytes)) : null,
        v.clock && !v.clock.reference && v.clock.offset !== undefined ? h("span", { class: "muted", title: "clock: reference = slope × local + offset" },
          `offset ${v.clock.offset >= 0 ? "+" : ""}${v.clock.offset.toFixed(3)} s`) : null,
        (pr.problems || []).length ? h("span", { class: "warn-text" }, pr.problems.join("; ")) : null);
      const mf = `${v.strip}|${v.playable}|${v.proxy}`;
      if (mf !== c.mediaFp && !players.includes(v.cam)) { c.mediaFp = mf; drawMedia(c); }
      const ctl = cameraTypeControl({ type: v.type, classes, disabled: v.excluded,
        onSave: (m, z) => saveType(v.cam, m, z), onAccept: () => acceptOne(v.cam) });
      c.ctl = ctl;
      replace(c.ctHost, ctl.el);
      replace(c.footHost,
        h("button", { class: "btn small", onclick: (e) => { e.stopPropagation(); narrateWith(v.cam); }, title: "Narration review with this camera (N)" }, "Narrate with this camera"),
        !v.playable && v.present ? h("button", { class: "btn small", onclick: (e) => { e.stopPropagation(); makeProxy(v.cam); },
          title: "H.264 proxy for playback here and in the narration tool" }, "Make proxy") : null,
        h("button", { class: "btn small ghost", onclick: (e) => { e.stopPropagation(); reprobe([v.cam]); } }, "Re-probe"),
        h("span", { class: "spacer" }),
        h("button", { class: "btn small ghost", onclick: (e) => { e.stopPropagation(); setExcluded(v, !v.excluded); } }, v.excluded ? "Use this video" : "Exclude"));
    }

    function drawMedia(c) {
      const v = c.v;
      if (!v.present) { replace(c.media, h("div", { class: "noplay" }, "The source file is missing.")); return; }
      const img = h("img", { class: "strip-img", src: v.strip + (v.strip.includes("h=") ? "" : "&h=96"), alt: `cam ${v.cam} frames`, loading: "lazy",
        title: v.playable ? "Click to play" : "",
        onerror: () => { img.replaceWith(h("div", { class: "noplay" }, "No thumbnails (ffmpeg could not read this file).")); } });
      img.addEventListener("click", (e) => { e.stopPropagation(); if (v.playable) play(v.cam); });
      replace(c.media, img,
        v.playable ? h("button", { class: "btn small play", onclick: (e) => { e.stopPropagation(); play(v.cam); } }, "▶ Play")
          : h("div", { class: "small muted", style: { padding: "4px 8px" } }, `${(v.probe || {}).codec || "this codec"}: not playable here — make a proxy`));
    }

    function play(cam) {
      const c = cards.get(cam);
      if (!c || !c.v.playable) return;
      if (players.includes(cam)) { const vid = c.media.querySelector("video"); if (vid) { vid.paused ? vid.play() : vid.pause(); } return; }
      while (players.length >= MAX_PLAYERS) {
        const old = cards.get(players.shift());
        if (old) { old.mediaFp = null; drawMedia(old); }
      }
      players.push(cam);
      const vid = h("video", { src: c.v.stream, controls: true, autoplay: true, preload: "metadata", playsinline: true });
      replace(c.media, vid, h("button", { class: "btn small play", style: { left: "auto", right: "8px", top: "8px", bottom: "auto" },
        onclick: (e) => { e.stopPropagation(); vid.pause(); vid.removeAttribute("src"); vid.load(); players.splice(players.indexOf(cam), 1); c.mediaFp = null; drawMedia(c); } }, "Close"));
    }

    // ---------------------------------------------------------------------------------------------- writes
    async function previewInvalidation(cam, mount, zoom) {
      try { return await put(`/api/takes/${seg(take)}/videos/${seg(cam)}/camera-type?preview=1`, { mount, zoom }); }
      catch { return null; }
    }

    async function confirmStale(title, prev) {
      const inval = (prev && prev.invalidated) || [];
      if (!inval.length) return true;
      const r = prev.rerun || {};
      return confirmDialog(title, [
        h("p", null, `This changes how the camera is solved, so ${inval.length} done stage${inval.length > 1 ? "s" : ""} become stale:`),
        h("div", { class: "dep-list" }, inval.map((id) => h("span", { class: "dep" }, h("span", { class: "st-g st-stale" }, "↻"), id))),
        r.p50_s ? h("p", null, `Re-running them takes about ${fmtDur(r.p50_s)}`, r.pod_usd_p50 ? ` and ${fmtUSD(r.pod_usd_p50)} of pod time (p50).` : ".") : null,
        h("p", { class: "muted" }, "Nothing re-runs by itself: you start it from Plan & run when you are ready.")],
      { okLabel: "Save and mark stale" });
    }

    async function saveType(cam, mount, zoom) {
      const label = CLASS_LABEL[classOf(mount, zoom)];
      const before = (videos.find((x) => x.cam === cam) || {}).type || {};
      const prev = await previewInvalidation(cam, mount, zoom);
      if (!await confirmStale(`Camera ${cam} → ${label}?`, prev)) { redrawCard(cam); return; }
      try {
        const res = await put(`/api/takes/${seg(take)}/videos/${seg(cam)}/camera-type`, { mount, zoom }, { ifMatch: etag });
        etag = res.toml_etag || etag;
        patchType(cam, res.type);
        // one keypress saves, so one click undoes: back to the earlier confirmed answer (same stale check)
        const undo = (before.state === "confirmed" || before.state === "conflict") && before.mount && typeof before.zoom === "boolean"
          && classOf(before.mount, before.zoom) !== classOf(mount, zoom)
          ? { label: "Undo", onClick: () => saveType(cam, before.mount, before.zoom) } : undefined;
        toast(`cam ${cam}: ${label} — saved to the take TOML` + (undo ? "" : " (H / T / Z change it)"), "good",
          { timeout: undo ? 8000 : 3500, action: undo });
        app.refreshNow();
      } catch (e) {
        await handleWriteError(e);
      }
    }

    async function acceptOne(cam) {
      const v = videos.find((x) => x.cam === cam);
      const sg = v && v.type && v.type.suggestion;
      if (!sg) return;
      const prev = await previewInvalidation(cam, sg.mount, sg.zoom);
      if (!await confirmStale(`Confirm cam ${cam} as ${CLASS_LABEL[classOf(sg.mount, sg.zoom)]}?`, prev)) return;
      await doAccept([cam], 0);
    }

    async function acceptMany(cams) {
      if (!cams.length) return;
      const all = new Set();
      let p50 = 0, usd = 0;
      for (const cam of cams) {
        const v = videos.find((x) => x.cam === cam);
        const sg = v.type.suggestion;
        const prev = await previewInvalidation(cam, sg.mount, sg.zoom);
        if (prev) { (prev.invalidated || []).forEach((x) => all.add(x)); p50 = Math.max(p50, (prev.rerun || {}).p50_s || 0); usd = Math.max(usd, (prev.rerun || {}).pod_usd_p50 || 0); }
      }
      const ok = await confirmStale(`Accept ${cams.length} suggestion${cams.length > 1 ? "s" : ""} (${cams.join(", ")})?`,
        { invalidated: Array.from(all), rerun: { p50_s: p50, pod_usd_p50: usd } });
      if (!ok) return;
      if (!all.size && !await confirmDialog("Accept suggestions?", `Cameras ${cams.join(", ")} take their suggested type (recorded as confirmed by you).`, { okLabel: "Accept" })) return;
      await doAccept(cams, 0.7);
    }

    async function doAccept(cams, minConf) {
      try {
        const res = await post(`/api/takes/${seg(take)}/camera-types/accept`, { cams, min_confidence: minConf }, { ifMatch: etag });
        etag = res.toml_etag || etag;
        for (const t of res.types || []) patchType(t.cam, t);
        toast(`Confirmed: ${cams.join(", ")}`, "good", { timeout: 3500 });
        app.refreshNow();
      } catch (e) { await handleWriteError(e); }
    }

    async function handleWriteError(e) {
      if (e.status === 412) {
        toast("The take TOML changed meanwhile (another tool or an editor): reloaded — please choose again.", "warn");
        await load(true);
      } else toast(errorText(e), "crit");
    }

    function patchType(cam, type) {
      const v = videos.find((x) => x.cam === cam);
      if (v && type) { v.type = { ...v.type, ...type }; const c = cards.get(cam); if (c) { c.fp = null; updateCard(c, v, c.i); } drawSummary(); }
    }

    function redrawCard(cam) { const c = cards.get(cam); if (c) { c.fp = null; updateCard(c, c.v, c.i); } }

    async function setExcluded(v, excluded) {
      const ok = await confirmDialog(excluded ? `Exclude cam ${v.cam}?` : `Use cam ${v.cam} again?`,
        excluded ? "The video is left out of every stage (written to the take TOML's exclude list). Stages that used it become stale."
          : "The video joins the camera set again; camera and body stages become stale.", { okLabel: excluded ? "Exclude" : "Use it" });
      if (!ok) return;
      try {
        const res = await patch(`/api/takes/${seg(take)}/videos/${seg(v.cam)}`, { excluded }, { ifMatch: etag });
        etag = res.toml_etag || etag;
        toast(excluded ? `cam ${v.cam} excluded` : `cam ${v.cam} used again`, "good");
        await load(true);
        app.refreshNow();
      } catch (e) { await handleWriteError(e); }
    }

    async function reprobe(cams) {
      try {
        const res = await post(`/api/takes/${seg(take)}/camera-types/suggest`, cams ? { cams } : {});
        toast(["Probing ", cams ? `cam ${cams.join(", ")}` : "every video", " in the background: ",
          h("a", { href: `#/runs/${res.run.id}` }, res.run.id), ". Suggestions appear here when it finishes."], "info");
      } catch (e) { toast(errorText(e), "crit"); }
    }

    async function makeProxy(cam) {
      try {
        const res = await post(`/api/takes/${seg(take)}/videos/${seg(cam)}/proxy`, {});
        toast(["Making an H.264 proxy for cam ", cam, ": ", h("a", { href: `#/runs/${res.run.id}` }, res.run.id)], "info");
      } catch (e) { toast(errorText(e), "crit"); }
    }

    async function narrateWith(cam) { (await app.actions()).narrate(app, take, cam); }

    // ---------------------------------------------------------------------------------------------- keys
    function select(i, scroll) {
      if (i < 0 || i >= videos.length) return;
      sel = i;
      for (const c of cards.values()) c.el.classList.toggle("sel", c.i === sel);
      const c = cards.get(videos[sel].cam);
      if (c && scroll) { c.el.scrollIntoView({ block: "nearest", behavior: "smooth" }); c.el.focus({ preventScroll: true }); }
    }

    // keyboard focus inside a card (Tab, a click on one of its buttons) selects that card, so the keys always act
    // on the card you are on, never on one selected earlier
    gridEl.addEventListener("focusin", (e) => {
      const cardEl = e.target && e.target.closest && e.target.closest(".vcard");
      if (!cardEl) return;
      const i = videos.findIndex((x) => x.cam === cardEl.dataset.cam);
      if (i >= 0 && i !== sel) select(i, false);
    });

    drawExplain();
    await load();
    return {
      refresh: () => load(false),
      onKey(e) {
        if (!videos.length) return false;
        const v = videos[sel];
        const c = v && cards.get(v.cam);
        const k = e.key;
        if (k === "j" || k === "J") { select(Math.min(videos.length - 1, sel + 1), true); return true; }
        if (k === "k" || k === "K") { select(Math.max(0, sel - 1), true); return true; }
        // a held key auto-repeats: only its first press may write (or start the narration tool)
        if (e.repeat && "hHtTzZaAnN ".includes(k.length === 1 ? k : "?")) return true;
        if (e.repeat && k === "Enter") return true;
        if (!c || v.excluded) return false;
        if (k === "h" || k === "H") { c.ctl.pick("mount", "handheld"); return true; }
        if (k === "t" || k === "T") { c.ctl.pick("mount", "tripod"); return true; }
        if (k === "z" || k === "Z") { c.ctl.toggleZoom(); return true; }
        if (k === "a" || k === "A" || k === "Enter") { c.ctl.confirm(); return true; }
        if (k === " ") { play(v.cam); return true; }
        if (k === "n" || k === "N") { narrateWith(v.cam); return true; }
        return false;
      },
      destroy() {
        for (const c of cards.values()) { const vid = c.media.querySelector("video"); if (vid) { vid.pause(); vid.removeAttribute("src"); vid.load(); } }
      },
    };
  },
};
