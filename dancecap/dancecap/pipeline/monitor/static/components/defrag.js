// The defrag map: one square per 1-s cell of the dance's final result, drawn on a canvas (display and PNG export share the
// same painter). complete = solid green; partial = amber with a hatch (denser = further along: front only < raw fit <
// refined awaiting stitch); none = dark and empty. Chunks: a coloured bar under each cell, a bar in the gutter where a chunk
// starts, a dotted outline on the cells that only belong to a chunk's overlap. Cells being worked on right now carry a
// travelling wave (static under prefers-reduced-motion). Every state also differs in shape / texture, never colour alone.
import { fmtClock } from "../util.js";

export const CHUNK_COLORS = ["#58a6ff", "#ff7eb6", "#4fd1c5", "#c4a1ff", "#ff9f6b", "#a3d977", "#f6d365", "#8ab4ff", "#f28b82", "#7fdbca"];
export const chunkColor = (i) => CHUNK_COLORS[((i % CHUNK_COLORS.length) + CHUNK_COLORS.length) % CHUNK_COLORS.length];

const GAP = 3;
const COL_STEPS = [30, 20, 15, 10, 6];
const HATCH = { front: 10, fit: 6, refined: 4 };

export function readTheme(el) {
  const cs = getComputedStyle(el);
  const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
  return {
    complete: v("--f-complete", "#2ea043"), completeEdge: v("--f-complete-edge", "#3fb950"),
    partial: v("--f-partial", "#f0a81c"), partialBg: v("--f-partial-bg", "#3a2f12"),
    none: v("--f-none", "#232321"), noneEdge: v("--f-none-edge", "#3a3a37"),
    run: v("--f-run", "#58a6ff"), text: v("--text", "#f4f3ef"), text2: v("--text-2", "#c3c2b7"), text3: v("--text-3", "#8f8e86"),
    bg: v("--surface-1", "#1a1a19"), border: v("--border", "#383835"), sel: v("--text", "#f4f3ef"),
    font: v("--font", "Segoe UI, sans-serif"),
  };
}

/** Geometry of the wrapped grid for a canvas CSS width W and n cells. */
export function layout(W, n) {
  const labelW = W < 560 ? 34 : 52;
  const avail = Math.max(60, W - labelW - 2);
  let cols = COL_STEPS[COL_STEPS.length - 1];
  for (const c of COL_STEPS) { if ((avail - GAP * (c - 1)) / c >= 22) { cols = c; break; } }
  cols = Math.max(1, Math.min(cols, n || 1));
  const cell = Math.max(10, Math.min(46, Math.floor((avail - GAP * (cols - 1)) / cols)));
  const rows = Math.max(1, Math.ceil((n || 1) / cols));
  return { W, labelW, cols, rows, cell, gap: GAP, top: 4, H: 4 + rows * (cell + GAP) };
}

export function cellRect(g, i) {
  const r = Math.floor(i / g.cols), c = i % g.cols;
  return { x: g.labelW + c * (g.cell + g.gap), y: g.top + r * (g.cell + g.gap), w: g.cell, h: g.cell };
}

export function hitCell(g, n, x, y) {
  const c = Math.floor((x - g.labelW) / (g.cell + g.gap)), r = Math.floor((y - g.top) / (g.cell + g.gap));
  if (c < 0 || c >= g.cols || r < 0) return -1;
  const rx = (x - g.labelW) - c * (g.cell + g.gap), ry = (y - g.top) - r * (g.cell + g.gap);
  if (rx > g.cell + g.gap / 2 || ry > g.cell + g.gap / 2) return -1;
  const i = r * g.cols + c;
  return i < n ? i : -1;
}

const patCache = new Map();
function hatchPattern(ctx, color, spacing, dpr) {
  const key = `${color}|${spacing}|${dpr}`;
  let p = patCache.get(key);
  if (p) return p;
  const s = Math.max(2, Math.round(spacing * dpr));
  const oc = document.createElement("canvas");
  oc.width = oc.height = s;
  const o = oc.getContext("2d");
  o.strokeStyle = color;
  o.lineWidth = Math.max(1, 1.4 * dpr);
  o.beginPath();
  o.moveTo(0, s); o.lineTo(s, 0);
  o.moveTo(-s / 2, s / 2); o.lineTo(s / 2, -s / 2);
  o.moveTo(s / 2, s * 1.5); o.lineTo(s * 1.5, s / 2);
  o.stroke();
  p = ctx.createPattern(oc, "repeat");
  if (p && p.setTransform && typeof DOMMatrix !== "undefined") p.setTransform(new DOMMatrix().scale(1 / dpr));
  patCache.set(key, p);
  return p;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y); ctx.lineTo(x + w - r, y); ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r); ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h); ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

/**
 * Paint the grid into ctx (already scaled to CSS pixels). model: {cells, chunkIndex: {id: n}}; t: seconds for the wave
 * (null = static); sel / hover: cell indexes or -1.
 */
export function paintMap(ctx, g, model, th, { t = null, sel = -1, hover = -1, dpr = 1 } = {}) {
  const cells = model.cells, idx = model.chunkIndex || {};
  const small = g.cell < 26;
  ctx.font = `${Math.max(9, Math.min(12, g.cell * 0.34))}px ${th.font}`;
  ctx.textBaseline = "top";
  // gutter: elapsed time at the start of each row
  ctx.fillStyle = th.text3;
  ctx.textAlign = "right";
  ctx.font = `11px ${th.font}`;
  for (let r = 0; r < g.rows; r++) {
    const i = r * g.cols;
    if (i >= cells.length) break;
    ctx.fillText(fmtClock(i * (model.cellS || 1)).replace(/\.0$/, ""), g.labelW - 8, g.top + r * (g.cell + g.gap) + 3);
  }
  ctx.font = `${Math.max(9, Math.min(12, g.cell * 0.34))}px ${th.font}`;
  for (let i = 0; i < cells.length; i++) {
    const c = cells[i], { x, y, w, h } = cellRect(g, i);
    const rad = Math.max(2, Math.round(g.cell * 0.1));
    // body
    if (c.s === "complete") {
      ctx.fillStyle = th.complete;
      roundRect(ctx, x, y, w, h, rad); ctx.fill();
    } else if (c.s === "partial") {
      ctx.fillStyle = th.partialBg;
      roundRect(ctx, x, y, w, h, rad); ctx.fill();
      ctx.save();
      roundRect(ctx, x, y, w, h, rad); ctx.clip();
      ctx.fillStyle = hatchPattern(ctx, th.partial, HATCH[c.g] || HATCH.front, dpr);
      ctx.translate(x, y);
      ctx.fillRect(0, 0, w, h);
      if (c.g === "refined" || c.g === "analysed") { ctx.fillStyle = th.complete; ctx.fillRect(0, 0, w, Math.max(2, Math.round(h * 0.12))); }
      ctx.restore();
      ctx.strokeStyle = th.partial; ctx.lineWidth = 1;
      roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, rad); ctx.stroke();
    } else {
      ctx.fillStyle = th.none;
      roundRect(ctx, x, y, w, h, rad); ctx.fill();
      ctx.strokeStyle = th.noneEdge; ctx.lineWidth = 1;
      roundRect(ctx, x + 0.5, y + 0.5, w - 1, h - 1, rad); ctx.stroke();
    }
    // the cameras that saw it (digit), a mark for mono / two-view / absent
    if (!small && c.n != null && c.s !== "none") {
      ctx.textAlign = "left";
      ctx.fillStyle = c.s === "complete" ? "rgba(255,255,255,.9)" : th.text;
      const tw = ctx.measureText(String(c.n)).width;
      ctx.save();
      ctx.globalAlpha = c.s === "complete" ? 0.85 : 1;
      if (c.s === "partial") { ctx.fillStyle = "rgba(0,0,0,.55)"; roundRect(ctx, x + 2, y + 2, tw + 6, g.cell * 0.34 + 3, 3); ctx.fill(); ctx.fillStyle = "#fff"; }
      ctx.fillText(String(c.n), x + 5, y + 3);
      ctx.restore();
    }
    if (c.s === "partial" && (c.f === "mono" || c.f === "two_view" || c.f === "absent")) {
      ctx.fillStyle = th.text;
      ctx.beginPath(); ctx.moveTo(x + w, y); ctx.lineTo(x + w - Math.max(7, g.cell * 0.28), y); ctx.lineTo(x + w, y + Math.max(7, g.cell * 0.28)); ctx.closePath();
      ctx.fillStyle = c.f === "absent" ? th.text3 : "#e5534b"; ctx.fill();
    }
    // chunk: bar under the cell; overlap: dotted outline in the neighbouring chunk's colour
    if (c.c != null && idx[c.c] != null) {
      ctx.fillStyle = chunkColor(idx[c.c]);
      ctx.fillRect(x + 1, y + h - 3, w - 2, 3);
    }
    if (c.o && c.o.length) {
      ctx.save();
      ctx.setLineDash([2, 2]);
      ctx.strokeStyle = chunkColor(idx[c.o[0]] ?? 0); ctx.lineWidth = 1.5;
      roundRect(ctx, x + 0.75, y + 0.75, w - 1.5, h - 1.5, rad); ctx.stroke();
      ctx.restore();
    }
    // being worked on: a wave along the chunk + a pulsing outline
    if (c.live) {
      const ph = t == null ? 0.5 : 0.5 + 0.5 * Math.sin(t * 3.4 - i * 0.42);
      ctx.save();
      ctx.globalAlpha = 0.10 + 0.30 * ph;
      ctx.fillStyle = "#ffffff";
      roundRect(ctx, x, y, w, h, rad); ctx.fill();
      ctx.restore();
      ctx.save();
      ctx.globalAlpha = 0.35 + 0.65 * ph;
      ctx.strokeStyle = th.run; ctx.lineWidth = 1.25;
      roundRect(ctx, x + 1, y + 1, w - 2, h - 2, rad); ctx.stroke();
      ctx.restore();
    }
    // chunk boundary: a bar in the gap before the first cell of a chunk run
    const prev = i > 0 ? cells[i - 1] : null;
    const startsRun = !prev || prev.c !== c.c;
    if (startsRun && c.c != null && idx[c.c] != null) {
      ctx.fillStyle = chunkColor(idx[c.c]);
      ctx.fillRect(x - 2, y, 2, h);
      if (g.cell >= 28) {
        const lab = String(c.c).slice(0, 4);
        const tw = ctx.measureText(lab).width + 8;
        ctx.fillStyle = chunkColor(idx[c.c]);
        roundRect(ctx, x + w - tw - 2, y + 2, tw, g.cell * 0.34 + 3, 3); ctx.fill();
        ctx.fillStyle = "#0b0b0b"; ctx.textAlign = "left";
        ctx.fillText(lab, x + w - tw + 2, y + 3);
      }
    } else if (startsRun && prev && prev.c != null) {
      ctx.fillStyle = th.text3; ctx.fillRect(x - 2, y, 2, h);
    }
    if (i === hover && i !== sel) {
      ctx.strokeStyle = th.text; ctx.lineWidth = 1;
      roundRect(ctx, x - 0.5, y - 0.5, w + 1, h + 1, rad + 1); ctx.stroke();
    }
    if (i === sel) {
      ctx.strokeStyle = th.sel; ctx.lineWidth = 2;
      roundRect(ctx, x - 1, y - 1, w + 2, h + 2, rad + 1); ctx.stroke();
    }
  }
}

/** The canvas controller: sizing, HiDPI, hover / selection / keyboard, the animation loop. */
export class Defrag {
  constructor(canvas, { onHover, onSelect } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onHover = onHover || (() => {});
    this.onSelect = onSelect || (() => {});
    this.model = { cells: [], chunkIndex: {}, cellS: 1 };
    this.sel = -1; this.hover = -1; this.g = layout(800, 1);
    this.raf = 0; this.last = 0;
    this.reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.ro = new ResizeObserver(() => this.resize());
    this.ro.observe(canvas.parentElement);
    canvas.addEventListener("pointermove", (e) => this._move(e));
    canvas.addEventListener("pointerleave", () => { this.hover = -1; this.onHover(-1); this.paint(); });
    canvas.addEventListener("click", (e) => { const i = this._at(e); this.select(i === this.sel ? -1 : i); });
    canvas.addEventListener("keydown", (e) => this._key(e));
    this._vis = () => this._loop(true);
    document.addEventListener("visibilitychange", this._vis);
    this.mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
    if (this.mq && this.mq.addEventListener) this.mq.addEventListener("change", () => this.paint());
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    this.ro.disconnect();
    document.removeEventListener("visibilitychange", this._vis);
  }

  setModel(model) {
    this.model = model;
    if (this.sel >= model.cells.length) this.sel = -1;
    this.resize();
  }

  resize() {
    const W = Math.max(280, Math.floor(this.canvas.parentElement.clientWidth || 800));
    this.g = layout(W, this.model.cells.length);
    const dpr = window.devicePixelRatio || 1;
    this.dpr = dpr;
    this.canvas.style.width = `${W}px`;
    this.canvas.style.height = `${this.g.H}px`;
    this.canvas.width = Math.round(W * dpr);
    this.canvas.height = Math.round(this.g.H * dpr);
    this.paint();
    this._loop(true);
  }

  anyLive() { return this.model.cells.some((c) => c.live); }

  _loop(kick) {
    cancelAnimationFrame(this.raf);
    if (!this.anyLive() || this.reduced || document.visibilityState !== "visible") return;
    const step = (ts) => {
      if (ts - this.last > 55 || kick) { this.last = ts; this.paint(ts / 1000); kick = false; }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }

  paint(t = null) {
    const { ctx, g } = this;
    const th = readTheme(this.canvas);
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, g.W, g.H);
    paintMap(ctx, g, this.model, th, { t: this.reduced ? null : t, sel: this.sel, hover: this.hover, dpr: this.dpr });
  }

  _at(e) {
    const r = this.canvas.getBoundingClientRect();
    return hitCell(this.g, this.model.cells.length, e.clientX - r.left, e.clientY - r.top);
  }

  _move(e) {
    const i = this._at(e);
    if (i !== this.hover) { this.hover = i; this.paint(); }
    this.onHover(i, e.clientX, e.clientY);
  }

  select(i) {
    this.sel = i;
    this.paint();
    this.onSelect(i);
  }

  _key(e) {
    const n = this.model.cells.length;
    if (!n) return;
    let i = this.sel < 0 ? 0 : this.sel;
    const k = e.key;
    if (k === "ArrowRight") i = Math.min(n - 1, i + 1);
    else if (k === "ArrowLeft") i = Math.max(0, i - 1);
    else if (k === "ArrowDown") i = Math.min(n - 1, i + this.g.cols);
    else if (k === "ArrowUp") i = Math.max(0, i - this.g.cols);
    else if (k === "Home") i = 0;
    else if (k === "End") i = n - 1;
    else if (k === "Escape") { this.select(-1); return; }
    else return;
    e.preventDefault();
    this.select(i);
  }

  /** The tooltip anchor of a cell in client coordinates. */
  anchorOf(i) {
    const r = this.canvas.getBoundingClientRect(), c = cellRect(this.g, i);
    return { x: r.left + c.x + c.w / 2, y: r.top + c.y + c.h };
  }

  /**
   * A PNG of the map with a header (title lines, the three numbers) and the chunk legend: drawn at 2x into an offscreen canvas.
   * head: {title, sub, stats: [{big, label, sub, color}], legend: [{color, text}], foot}
   */
  async exportBlob(head) {
    const W = 1240, scale = 2;
    const th = readTheme(this.canvas);
    const g = layout(W - 48, this.model.cells.length);
    const headH = 150, legendRows = Math.ceil(((head.legend || []).length + 5) / 3), legH = 70 + legendRows * 22;
    const H = headH + g.H + legH + 36;
    const oc = document.createElement("canvas");
    oc.width = W * scale; oc.height = H * scale;
    const ctx = oc.getContext("2d");
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    ctx.fillStyle = th.bg; ctx.fillRect(0, 0, W, H);
    ctx.textBaseline = "top"; ctx.textAlign = "left";
    ctx.fillStyle = th.text; ctx.font = `700 24px ${th.font}`; ctx.fillText(head.title, 24, 18);
    ctx.fillStyle = th.text3; ctx.font = `13px ${th.font}`; ctx.fillText(head.sub || "", 24, 50);
    let x = 24;
    for (const s of head.stats || []) {
      ctx.fillStyle = s.color; ctx.font = `700 44px ${th.font}`; ctx.fillText(s.big, x, 70);
      const bw = ctx.measureText(s.big).width;
      ctx.fillStyle = th.text; ctx.font = `600 15px ${th.font}`; ctx.fillText(s.label, x, 120);
      const lw = ctx.measureText(s.label).width;
      ctx.fillStyle = th.text3; ctx.font = `13px ${th.font}`; ctx.fillText(s.sub, x + lw + 10, 121);
      x += Math.max(bw, 200) + 80;
    }
    ctx.save();
    ctx.translate(24, headH);
    paintMap(ctx, g, this.model, th, { t: null, sel: -1, hover: -1, dpr: scale });
    ctx.restore();
    let y = headH + g.H + 18;
    ctx.font = `13px ${th.font}`;
    const key = [
      ["complete", () => { ctx.fillStyle = th.complete; ctx.fillRect(0, 0, 16, 16); }],
      ["partial: front only", () => hatchSwatch(ctx, th, "front", scale)],
      ["partial: raw fit", () => hatchSwatch(ctx, th, "fit", scale)],
      ["partial: refined, awaiting stitch", () => hatchSwatch(ctx, th, "refined", scale)],
      ["not done", () => { ctx.fillStyle = th.none; ctx.fillRect(0, 0, 16, 16); ctx.strokeStyle = th.noneEdge; ctx.strokeRect(0.5, 0.5, 15, 15); }],
    ];
    let kx = 24;
    for (const [label, sw] of key) {
      ctx.save(); ctx.translate(kx, y); sw(); ctx.restore();
      ctx.fillStyle = th.text2; ctx.fillText(label, kx + 22, y + 1);
      kx += 22 + ctx.measureText(label).width + 26;
    }
    y += 26;
    ctx.fillStyle = th.text3; ctx.fillText("digit = cameras that saw the second   bar under a cell = chunk core   dotted outline = chunk overlap   blue outline = being worked on now", 24, y);
    y += 24;
    const colW = (W - 48) / 3;
    (head.legend || []).forEach((l, k) => {
      const lx = 24 + (k % 3) * colW, ly = y + Math.floor(k / 3) * 22;
      ctx.fillStyle = l.color; ctx.fillRect(lx, ly + 3, 26, 5);
      ctx.fillStyle = th.text2; ctx.fillText(l.text, lx + 34, ly);
    });
    ctx.fillStyle = th.text3; ctx.font = `12px ${th.font}`;
    ctx.fillText(head.foot || "", 24, H - 22);
    return new Promise((res) => oc.toBlob((b) => res(b), "image/png"));
  }
}

function hatchSwatch(ctx, th, stage, dpr) {
  ctx.fillStyle = th.partialBg; ctx.fillRect(0, 0, 16, 16);
  ctx.save(); ctx.beginPath(); ctx.rect(0, 0, 16, 16); ctx.clip();
  ctx.fillStyle = hatchPattern(ctx, th.partial, HATCH[stage], dpr); ctx.fillRect(0, 0, 16, 16);
  if (stage === "refined") { ctx.fillStyle = th.complete; ctx.fillRect(0, 0, 16, 3); }
  ctx.restore();
  ctx.strokeStyle = th.partial; ctx.strokeRect(0.5, 0.5, 15, 15);
}
