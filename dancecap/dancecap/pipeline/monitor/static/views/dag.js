// Pipeline DAG (PIPELINE_MONITOR.md 6.5): 8 phase columns, rows in dependency order (barycentre sweep), SVG edges,
// tinted bands for pod sessions, the critical path in bold, dashed loop back-edges, per-node status + progress +
// ETA + where it runs. Hovering a node highlights its dependencies; hovering an edge says why the order is so.
import { s, h, replace, fmtDur, fmtEta, pct } from "../util.js";
import { statusMeta, WHERE } from "../components/status.js";

const NODE_W = 176, NODE_H = 52, COL_W = 212, ROW_H = 72, TOP = 46, LEFT = 30;
const HIDE = new Set(["skipped", "n/a"]);

function truncate(t, n) { return t.length > n ? t.slice(0, n - 1) + "…" : t; }

function subline(st) {
  const p = st.progress;
  if (st.status === "running") {
    const frac = p && (p.frac ?? (p.total ? p.done / p.total : null));
    const left = frac !== null && frac !== undefined ? pct(frac) : "running";
    return `${left}${st.eta ? " · ETA " + fmtDur(st.eta.p50_s, { short: true }) : ""}`;
  }
  if (st.status === "queued") return `queued · ${fmtDur(st.estimate && st.estimate.p50_s, { short: true })}`;
  if (st.status === "done") return st.session ? `done · ${st.session}` : "done";
  if (st.status === "planned") return "not built yet";
  if (st.status === "stale") return "stale: " + ((st.outputs && st.outputs.upstream_stale && st.outputs.upstream_stale[0]) ? "after " + st.outputs.upstream_stale[0] : "re-run");
  if (st.status === "blocked") return "needs you";
  if (st.status === "failed") return "failed" + (st.last_run && st.last_run.rc !== undefined ? ` (rc ${st.last_run.rc})` : "");
  const e = st.estimate || {};
  return `${statusMeta(st.status).label} · ~${fmtDur(e.p50_s, { short: true })}`;
}

function nodeTitle(st) {
  const lines = [`${st.id}: ${st.name || ""}`, `status: ${statusMeta(st.status).label}${st.status_reason ? " (" + st.status_reason + ")" : ""}`,
    `runs on: ${(WHERE[st.runs_on] || {}).label || st.runs_on}${st.session ? " · session " + st.session : ""}`];
  if (st.estimate) lines.push(`estimate: ${fmtEta(st.estimate)}${st.estimate.cost_p50_usd ? ` · $${st.estimate.cost_p50_usd.toFixed(2)}` : ""} (${st.estimate.confidence || "?"})`);
  if (st.camera_types) {
    const ct = Object.entries(st.camera_types).filter(([, v]) => v && v.length).map(([k, v]) => `${k}: ${v.join(" ")}`);
    if (ct.length) lines.push("camera types: " + ct.join(" | "));
  }
  return lines.join("\n");
}

/** Order nodes inside each column: dependency order first, barycentre of upstream rows as the priority. */
function layout(stages, edges) {
  const byId = new Map(stages.map((st) => [st.id, st]));
  const cols = new Map();
  stages.forEach((st, i) => { st._i = i; if (!cols.has(st.column)) cols.set(st.column, []); cols.get(st.column).push(st); });
  const preds = new Map(stages.map((st) => [st.id, []]));
  for (const [a, b, kind] of edges) {
    if (!byId.has(a) || !byId.has(b) || String(kind).startsWith("loop")) continue;
    preds.get(b).push(a);
  }
  const row = new Map();
  const colIds = Array.from(cols.keys()).sort((a, b) => a - b);
  for (let pass = 0; pass < 2; pass++) {
    for (const c of colIds) {
      const list = cols.get(c);
      const inCol = new Set(list.map((x) => x.id));
      const indeg = new Map(list.map((x) => [x.id, 0]));
      for (const x of list) for (const p of preds.get(x.id)) if (inCol.has(p)) indeg.set(x.id, indeg.get(x.id) + 1);
      const bary = (x) => {
        const ps = preds.get(x.id).filter((p) => !inCol.has(p) && row.has(p));
        const own = pass > 0 && row.has(x.id) ? row.get(x.id) : x._i / 100;
        return ps.length ? ps.reduce((a, p) => a + row.get(p), 0) / ps.length : own;
      };
      const ordered = [];
      const ready = list.filter((x) => indeg.get(x.id) === 0);
      while (ready.length) {
        ready.sort((a, b) => bary(a) - bary(b) || (a.session || "~").localeCompare(b.session || "~") || a._i - b._i);
        const x = ready.shift();
        ordered.push(x);
        for (const y of list) {
          if (preds.get(y.id).includes(x.id)) {
            indeg.set(y.id, indeg.get(y.id) - 1);
            if (indeg.get(y.id) === 0) ready.push(y);
          }
        }
      }
      for (const x of list) if (!ordered.includes(x)) ordered.push(x); // cycle guard
      ordered.forEach((x, i) => row.set(x.id, i));
      cols.set(c, ordered);
    }
  }
  const pos = new Map();
  let maxRows = 0;
  for (const c of colIds) {
    const list = cols.get(c);
    maxRows = Math.max(maxRows, list.length);
    list.forEach((x, i) => pos.set(x.id, { x: LEFT + c * COL_W, y: TOP + i * ROW_H, col: c, row: i }));
  }
  return { pos, cols, colIds, maxRows };
}

/**
 * renderDag(host, snap, {registry, selected, showAll, onSelect}) -> {highlight(id)}
 * host is a .dag-wrap element; its scroll position survives re-renders.
 */
export function renderDag(host, snap, { registry = null, selected = null, showAll = false, fit = false, onSelect } = {}) {
  const stages = (snap.stages || []).filter((st) => st.in_variant !== false && (showAll || !HIDE.has(st.status)));
  const ids = new Set(stages.map((st) => st.id));
  const edges = (snap.edges || []).filter(([a, b]) => ids.has(a) && ids.has(b));
  const { pos, cols, colIds, maxRows } = layout(stages, edges);
  const ncol = Math.max(8, (snap.columns || []).length);
  const W = LEFT + ncol * COL_W + 10;
  const H = TOP + maxRows * ROW_H + 50;
  const svg = s("svg", { width: fit ? "100%" : W, height: fit ? null : H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Pipeline of ${snap.take}: ${stages.length} stages` });
  svg.appendChild(s("defs", null,
    s("marker", { id: "arr", viewBox: "0 0 8 8", refX: "7", refY: "4", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse" },
      s("path", { d: "M0,0 L8,4 L0,8 z", fill: "context-stroke", style: "fill: var(--text-3)" }))));

  // column backgrounds + labels
  (snap.columns || []).forEach((label, c) => {
    if (c % 2 === 0) svg.appendChild(s("rect", { class: "dag-col-bg", x: LEFT + c * COL_W - 14, y: 0, width: COL_W, height: H, opacity: 0.5 }));
    svg.appendChild(s("text", { class: "dag-col-label", x: LEFT + c * COL_W, y: 22 }, String(label).replace(/^\d+\s*/, (m) => m.trim() + "  ")));
  });

  // session bands: consecutive rows of the same session in one column become one rect
  const bandG = s("g");
  const labelled = new Set();
  const sid = new Map((snap.groups || []).map((g) => [g.id, g.sid]));
  for (const c of colIds) {
    const list = cols.get(c);
    let i = 0;
    while (i < list.length) {
      const g = list[i].session;
      if (!g) { i++; continue; }
      let j = i;
      while (j + 1 < list.length && list[j + 1].session === g) j++;
      const p0 = pos.get(list[i].id), p1 = pos.get(list[j].id);
      const cls = ["S1_front", "S2_fit", "S_room", "S_splats"].includes(g) ? g : "other";
      bandG.appendChild(s("rect", { class: "band " + cls, x: p0.x - 7, y: p0.y - 7, width: NODE_W + 14, height: p1.y - p0.y + NODE_H + 14, rx: 9 }));
      if (!labelled.has(g)) {
        labelled.add(g);
        bandG.appendChild(s("text", { class: "band-label " + cls, x: p0.x - 4, y: p0.y - 10 },
          g + (sid.get(g) ? " · " + sid.get(g) : "")));
      }
      i = j + 1;
    }
  }
  svg.appendChild(bandG);

  // edges
  const crit = new Set();
  const cp = snap.critical_path || [];
  for (let i = 0; i + 1 < cp.length; i++) crit.add(cp[i] + ">" + cp[i + 1]);
  const helps = new Map();
  const loops = new Map();
  if (registry) {
    for (const st of registry.stages || []) for (const hp of st.helps || []) helps.set(st.id + ">" + hp.stage, hp.why);
    for (const lp of registry.loops || []) loops.set(lp.id, lp);
  }
  const edgeG = s("g");
  const edgeEls = [];
  for (const [a, b, kind] of edges) {
    const pa = pos.get(a), pb = pos.get(b);
    let d;
    const isLoop = String(kind).startsWith("loop");
    if (isLoop || pa.col > pb.col) {
      const yb = H - 16 - (edgeEls.length % 3) * 6;
      d = `M${pa.x + NODE_W / 2},${pa.y + NODE_H} C${pa.x + NODE_W / 2},${yb} ${pb.x + NODE_W / 2},${yb} ${pb.x + NODE_W / 2},${pb.y + NODE_H}`;
    } else if (pa.col === pb.col) {
      const bulge = 9 + 5 * Math.min(Math.abs(pb.row - pa.row), 4);
      const x = pa.x;
      d = `M${x},${pa.y + NODE_H / 2 + 6} C${x - bulge},${pa.y + NODE_H / 2 + 6} ${x - bulge},${pb.y + NODE_H / 2 - 6} ${x},${pb.y + NODE_H / 2 - 6}`;
    } else {
      const x1 = pa.x + NODE_W, y1 = pa.y + NODE_H / 2, x2 = pb.x, y2 = pb.y + NODE_H / 2;
      const dx = Math.max(30, (x2 - x1) * 0.5);
      d = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
    }
    const cls = ["edge", kind === "soft" ? "soft" : "", isLoop ? "loop" : "", crit.has(a + ">" + b) ? "crit" : ""];
    const path = s("path", { class: cls, d, "marker-end": "url(#arr)", "data-a": a, "data-b": b });
    let why = helps.get(a + ">" + b);
    if (isLoop) {
      const lp = loops.get(kind);
      why = `loop ${kind}${lp ? ": " + (lp.when || "") : ""}`;
    } else if (!why) why = kind === "soft" ? `${b} uses ${a} when it exists (soft dependency)` : `${b} needs ${a}'s outputs`;
    const hit = s("path", { class: "edge-hit", d, "data-a": a, "data-b": b, "data-why": `${a} → ${b}: ${why}` });
    edgeEls.push(path);
    edgeG.appendChild(path);
    edgeG.appendChild(hit);
  }
  svg.appendChild(edgeG);

  // nodes
  const nodeG = s("g");
  const cpSet = new Set(cp);
  for (const st of stages) {
    const p = pos.get(st.id);
    const m = statusMeta(st.status);
    const g = s("g", {
      class: ["node", "st-" + String(st.status).replace("/", ""), "where-" + st.runs_on, st.id === selected ? "sel" : "",
        cpSet.has(st.id) ? "critical" : "", st.optional ? "optional" : ""].join(" "),
      transform: `translate(${p.x},${p.y})`, tabindex: "0", role: "button", "data-id": st.id,
      "aria-label": `${st.id}, ${m.label}`,
    });
    g.appendChild(s("title", null, nodeTitle(st)));
    g.appendChild(s("rect", { class: "n-box", width: NODE_W, height: NODE_H, rx: 7 }));
    g.appendChild(s("rect", { class: "n-badge", x: 8, y: 8, width: 18, height: 18, rx: 4 }));
    g.appendChild(s("text", { class: "n-badge-t", x: 17, y: 21, "text-anchor": "middle" }, (WHERE[st.runs_on] || { badge: "?" }).badge));
    g.appendChild(s("text", { class: "n-id", x: 33, y: 21 }, truncate(st.id, 19)));
    g.appendChild(s("circle", { class: "n-glyph-c", cx: NODE_W - 15, cy: 17, r: 8 }));
    g.appendChild(s("text", { class: "n-glyph-t", x: NODE_W - 15, y: 17.5 }, m.glyph));
    g.appendChild(s("text", { class: "n-sub", x: 9, y: 42 }, truncate(subline(st), 29)));
    if (st.status === "running") {
      const pr = st.progress || {};
      const frac = pr.frac ?? (pr.total ? pr.done / pr.total : null);
      g.appendChild(s("rect", { class: "n-pbar-bg", x: 1, y: NODE_H - 5, width: NODE_W - 2, height: 4, rx: 2 }));
      if (frac !== null && frac !== undefined) g.appendChild(s("rect", { class: "n-pbar", x: 1, y: NODE_H - 5, width: Math.max(2, (NODE_W - 2) * Math.min(1, frac)), height: 4, rx: 2 }));
    }
    g.addEventListener("click", () => onSelect && onSelect(st.id));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect && onSelect(st.id); } });
    nodeG.appendChild(g);
  }
  svg.appendChild(nodeG);

  replace(host, svg);

  // hover: highlight a node's in/out edges (and their nodes)
  let tip = null;
  function hl(id) {
    host.classList.toggle("hovering", !!id);
    for (const e of edgeEls) {
      const on = id && (e.dataset.a === id || e.dataset.b === id);
      e.classList.toggle("hl", !!on);
      e.classList.toggle("out", !!on && e.dataset.a === id);
    }
    for (const n of nodeG.children) {
      const nid = n.dataset.id;
      n.classList.toggle("hl", !!id && (nid === id || edgeEls.some((e) => e.classList.contains("hl") && (e.dataset.a === nid || e.dataset.b === nid))));
    }
  }
  nodeG.addEventListener("mouseover", (e) => { const g = e.target.closest(".node"); if (g) hl(g.dataset.id); });
  nodeG.addEventListener("mouseleave", () => hl(null));
  edgeG.addEventListener("mousemove", (e) => {
    const t = e.target.closest(".edge-hit");
    if (!t) return;
    if (!tip) { tip = h("div", { class: "dag-tip" }); document.body.appendChild(tip); }
    tip.textContent = t.dataset.why;
    tip.style.left = Math.min(e.clientX + 14, window.innerWidth - 400) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    hl(null);
    for (const ed of edgeEls) if (ed.dataset.a === t.dataset.a && ed.dataset.b === t.dataset.b) { ed.classList.add("hl"); host.classList.add("hovering"); }
  });
  edgeG.addEventListener("mouseleave", () => { if (tip) { tip.remove(); tip = null; } hl(null); });
  return { highlight: hl, destroy() { if (tip) tip.remove(); } };
}
