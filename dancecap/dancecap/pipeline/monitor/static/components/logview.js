// Log viewer over LogChunk endpoints ({offset, next, size, eof, lines, truncated_head}). Lines arrive redacted by
// the server and are rendered as text only. "Follow" polls every second and keeps the view at the bottom.
import { h, replace, debounce } from "../util.js";
import { get } from "../api.js";

const MAX_LINES = 6000;
const BAD = /\b(error|failed|failure|traceback|exception|killed|oom)\b/i;
const WARN = /\b(warn(ing)?|stale|retry|slow)\b/i;

export function logView({ url, follow = true, height = null, title = null, showControls = true } = {}) {
  let src = url;
  let next = null;
  let timer = null;
  let lines = [];
  let filter = "";
  let following = follow;
  let alive = true;
  const body = h("div", { class: "log-body", tabindex: "0", style: height ? { height } : null });
  const info = h("span", { class: "muted small" });
  const followBox = h("input", { type: "checkbox", checked: following });
  const filterBox = h("input", { type: "search", placeholder: "filter", class: "small-input", "aria-label": "Filter lines" });
  followBox.addEventListener("change", () => { following = followBox.checked; if (following) { scrollEnd(); tick(); } });
  filterBox.addEventListener("input", debounce(() => { filter = filterBox.value.toLowerCase(); render(); }, 150));
  body.addEventListener("scroll", () => {
    const atEnd = body.scrollHeight - body.scrollTop - body.clientHeight < 24;
    if (!atEnd && following && document.activeElement === body) { following = false; followBox.checked = false; }
  });
  const el = h("div", { class: "logview" },
    showControls ? h("div", { class: "log-bar" },
      title ? h("strong", null, title) : null, filterBox,
      h("label", { class: "chk" }, followBox, "follow"), info) : null,
    body);

  function lineEl(t) {
    return h("div", { class: ["ll", BAD.test(t) ? "ll-bad" : WARN.test(t) ? "ll-warn" : ""] }, t);
  }
  function render() {
    const shown = filter ? lines.filter((l) => l.toLowerCase().includes(filter)) : lines;
    replace(body, shown.length ? shown.map(lineEl) : h("div", { class: "muted ll" }, src ? "(empty)" : "(no source)"));
    if (following) scrollEnd();
  }
  function scrollEnd() { body.scrollTop = body.scrollHeight; }

  async function tick() {
    if (!alive || !src) return;
    try {
      const q = next === null ? "tail=1&max_bytes=131072" : `offset=${next}&max_bytes=65536`;
      const c = await get(src + (src.includes("?") ? "&" : "?") + q);
      if (!alive) return;
      const fresh = c.lines || [];
      if (next !== null && c.offset < next) { lines = []; } // file rotated or truncated
      next = c.next ?? next;
      if (fresh.length) {
        const wasEmpty = lines.length === 0;
        lines = lines.concat(fresh);
        if (lines.length > MAX_LINES) lines = lines.slice(-MAX_LINES);
        if (wasEmpty || filter) render();
        else {
          const kids = fresh.filter((l) => !filter || l.toLowerCase().includes(filter)).map(lineEl);
          kids.forEach((k) => body.appendChild(k));
          while (body.childNodes.length > MAX_LINES) body.removeChild(body.firstChild);
          if (following) scrollEnd();
        }
      } else if (lines.length === 0) render();
      info.textContent = c.size !== undefined ? `${Math.round(c.size / 1024)} KB${c.truncated_head ? " (tail)" : ""}` : "";
    } catch (e) {
      info.textContent = e.status === 404 ? "no log yet" : "log unavailable";
      if (lines.length === 0) render();
    }
  }
  function loop() {
    clearTimeout(timer);
    if (!alive) return;
    timer = setTimeout(async () => {
      if (following && document.visibilityState === "visible") await tick();
      loop();
    }, 1000);
  }

  tick().then(loop);
  return {
    el,
    setSource(u) { src = u; next = null; lines = []; render(); tick(); },
    refresh: tick,
    destroy() { alive = false; clearTimeout(timer); },
  };
}
