// Move narration tool (dancecap.moves.annotate). Plain JS, no build step, served from 127.0.0.1 only.
// Times on the timeline are LABEL times: the take's reference clock (take mode) or the video's own clock.
'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video');
const canvas = $('timeline');
const MIN_LEN = 0.1;
const FAMILIES = ['rest', 'hold', 'step', 'open', 'turn', 'head', 'wave'];
const TONES = { green: '#4caf50', blue: '#4052b5', yellow: '#ffeb3b', orange: '#ff9800' };
const TL = { ruler: [0, 16], beats: [16, 32], cb: [33, 39], segs: [41, 101], words: [104, 156] };

const S = {
  info: null, moves: [], moveBy: {}, cams: {}, cam: null, segs: [], sel: null, dirty: false,
  undo: [], redo: [], view: null, range: [0, 60], grid: [], proposals: null, cb: [],
  narr: null, words: [], spans: [], taps: [], loop: null, rec: null, drag: null, hover: null,
  settings: { lag: 0.4, snap: 'half', engine: 'auto', typesafe: false, speed: 0.5 },
  switching: false, needDraw: true, lastSaved: null, pick: { items: [], idx: 0 }, filter: 'all',
};

// ------------------------------------------------------------------------------------------------ utilities
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clone = (x) => JSON.parse(JSON.stringify(x));
const uid = (p) => `${p}-${Math.random().toString(16).slice(2, 12)}`;
const nowIso = () => new Date().toISOString().replace(/\.\d+Z$/, '+00:00');

function fmt(t) {
  if (t == null || !isFinite(t)) return '-';
  const s = Math.abs(t), m = Math.floor(s / 60), r = s - 60 * m;
  return (t < 0 ? '-' : '') + m + ':' + r.toFixed(2).padStart(5, '0');
}
function toast(msg, err) {
  const d = document.createElement('div');
  d.textContent = msg;
  if (err) d.className = 'err';
  $('toast').appendChild(d);
  setTimeout(() => d.remove(), err ? 9000 : 3500);
}
window.addEventListener('error', (e) => toast('UI error: ' + e.message, true));
window.addEventListener('unhandledrejection', (e) => toast('Error: ' + ((e.reason && e.reason.message) || e.reason), true));

async function api(method, url, body, form) {
  const opt = { method, headers: {} };
  if (body !== undefined) {
    if (form) opt.body = body;
    else { opt.headers['Content-Type'] = 'application/json'; opt.body = JSON.stringify(body); }
  }
  const r = await fetch(url, opt);
  let d = null;
  try { d = await r.json(); } catch (e) { d = null; }
  if (!r.ok && r.status !== 202) {
    const msg = (d && (d.error || (d.problems || []).join('\n'))) || `${r.status} ${r.statusText}`;
    const err = new Error(msg); err.data = d; err.status = r.status; throw err;
  }
  return d;
}
async function pollJob(jid, onTick) {
  for (;;) {
    const j = await api('GET', '/api/jobs/' + jid);
    if (onTick) onTick(j);
    if (j.state !== 'running') return j;
    await sleep(600);
  }
}
function store(k, v) { try { localStorage.setItem('dcannot.' + k, JSON.stringify(v)); } catch (e) { /* ignore */ } }
function recall(k, d) { try { const v = localStorage.getItem('dcannot.' + k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } }

// ------------------------------------------------------------------------------------------------ clocks
const camObj = (id) => S.cams[id || S.cam];
const toLabel = (v, id) => { const c = camObj(id); return c.slope * v + c.offset; };
const toVideo = (t, id) => { const c = camObj(id); return (t - c.offset) / c.slope; };
const nowT = () => toLabel(video.currentTime || 0);
function seekT(t) {
  const d = isFinite(video.duration) ? video.duration : 1e9;
  video.currentTime = clamp(toVideo(t), 0, Math.max(0, d - 0.02));
  S.needDraw = true;
}
function gridTimes(level) {
  return S.grid.filter((g) => level === 'measure' ? g[2] === 0 : level === 'half' ? g[2] % 4 === 0 : true).map((g) => g[0]);
}
function measureOf(t) {
  if (!S.grid.length) return '';
  let m = 0, last = null;
  for (const g of S.grid) { if (g[0] > t + 1e-6) break; if (g[2] === 0) m++; last = g; }
  if (!last) return '';
  return `m${m} . ${Math.floor(last[2] / 2) + 1}${last[2] % 2 ? '&' : ''}`;
}

// ------------------------------------------------------------------------------------------------ moves
function setMoves(list) {
  S.moves = list || [];
  S.moveBy = {};
  for (const m of S.moves) S.moveBy[m.slug] = m;
}
const moveName = (slug) => (S.moveBy[slug] && S.moveBy[slug].name) || slug;
function moveColor(seg) {
  if (!seg.move) return '#c0392b';
  const m = S.moveBy[seg.move];
  return TONES[(m && m.tone) || 'green'] || '#4caf50';
}
function norm(s) { return String(s || '').normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase(); }
function searchMoves(q) {
  q = norm(q).trim();
  const out = [];
  for (const m of S.moves) {
    const names = [m.name, m.slug.replace(/-/g, ' '), m.name_en, ...(m.aliases || [])].filter(Boolean).map(norm);
    let best = q ? 0 : 0.1;
    for (const n of names) {
      if (!q) break;
      if (n === q) best = Math.max(best, 5);
      else if (n.startsWith(q)) best = Math.max(best, 4);
      else if (n.split(/\s+/).some((w) => w.startsWith(q))) best = Math.max(best, 3);
      else if (n.includes(q)) best = Math.max(best, 2);
      else {
        let i = 0;
        for (const ch of n) { if (ch === q[i]) i++; if (i === q.length) break; }
        if (i === q.length) best = Math.max(best, 1);
      }
    }
    if (best > 0) out.push({ m, s: best + (m.status === 'scaffold' ? 0.05 : 0) });
  }
  out.sort((a, b) => b.s - a.s || a.m.name.localeCompare(b.m.name));
  return out.slice(0, 14).map((x) => x.m);
}

// ------------------------------------------------------------------------------------------------ segments
function sortSegs() { S.segs.sort((a, b) => a.t0 - b.t0 || a.t1 - b.t1); }
function selSeg() { return S.segs.find((s) => s.id === S.sel) || null; }
function segAt(t) { return S.segs.find((s) => s.t0 <= t + 1e-9 && t < s.t1) || null; }
function snapshot() { S.undo.push(clone(S.segs)); if (S.undo.length > 300) S.undo.shift(); S.redo = []; }
function changed() { sortSegs(); S.dirty = true; S.needDraw = true; renderAll(); scheduleDraft(); }
function undo() { if (!S.undo.length) return; S.redo.push(clone(S.segs)); S.segs = S.undo.pop(); changed(); }
function redo() { if (!S.redo.length) return; S.undo.push(clone(S.segs)); S.segs = S.redo.pop(); changed(); }
function humanise(seg) {
  if (seg.move) seg.provenance = 'human';
  seg.confidence = null;
  seg.annotator = S.info.annotator;
}
let draftTimer = null;
function scheduleDraft() {
  clearTimeout(draftTimer);
  draftTimer = setTimeout(() => { api('POST', '/api/draft', { segments: S.segs }).catch(() => {}); }, 2000);
}
function needsReview(s) { return !s.move || s.provenance === 'narration' || s.provenance === 'auto' || (s.confidence != null && s.confidence < 0.4); }

function assignMove(slug) {
  const seg = selSeg();
  if (!seg) return;
  snapshot();
  if (seg.move && seg.move !== slug && seg.provenance === 'narration') seg.relabelled_from = seg.move;
  seg.move = slug;
  const heard = seg.unresolved || seg.anchor_word;
  delete seg.unresolved; delete seg.candidates;
  if (heard) seg.heard = heard;
  humanise(seg);
  changed();
  maybeOfferAlias(seg, heard);
}
function maybeOfferAlias(seg, heard) {
  if (!heard) return;
  api('POST', '/api/suggest', { phrase: heard }).then((c) => {
    // offer to remember the phrase only when it names no move exactly yet (an alias of another move must be
    // changed in taxonomy/additions.toml by hand)
    const exact = c.find((x) => x.score >= 100 && x.move);
    seg._teach = exact ? null : heard;
    renderEditor();
  }).catch(() => {});
}
function splitAt(t, seg) {
  seg = seg || segAt(t);
  if (!seg || t <= seg.t0 + MIN_LEN || t >= seg.t1 - MIN_LEN) return false;
  snapshot();
  const b = clone(seg);
  b.id = uid('seg'); b.t0 = t; b.created = nowIso();
  seg.t1 = t;
  humanise(seg); humanise(b);
  S.segs.push(b);
  S.sel = b.id;
  changed();
  return true;
}
function mark() {
  const t = snapToGrid(nowT(), 0.12);
  if (S.rec) { recLog('tap'); S.taps.push(t); S.needDraw = true; toast('tap ' + fmt(t)); return; }
  if (splitAt(t)) return;
  // in a gap: a new segment up to the next one (or one measure), move to be chosen
  const next = S.segs.filter((s) => s.t0 > t).sort((a, b) => a.t0 - b.t0)[0];
  const m = gridTimes('measure').find((g) => g > t + 0.5);
  const t1 = Math.min(next ? next.t0 : Infinity, m != null ? m : t + 3, S.range[1]);
  if (t1 - t < MIN_LEN) return;
  snapshot();
  const s = { id: uid('seg'), t0: t, t1, move: null, node: null, tier: 'couple', provenance: 'human', confidence: null,
    annotator: S.info.annotator, transcript: null, notes: null, created: nowIso(), unresolved: '' };
  S.segs.push(s); S.sel = s.id; changed(); focusPicker();
}
function snapToGrid(t, tol) {
  let best = null;
  for (const g of S.grid) { const d = Math.abs(g[0] - t); if (d <= tol && (best == null || d < Math.abs(best - t))) best = g[0]; }
  return best == null ? t : best;
}
function splitSelected() { const s = selSeg(); const t = nowT(); if (s && splitAt(t, s)) return; mark(); }
function mergeNext() {
  const s = selSeg(); if (!s) return;
  const next = S.segs.filter((x) => x.t0 >= s.t1 - 1e-6 && x.id !== s.id).sort((a, b) => a.t0 - b.t0)[0];
  if (!next) return;
  snapshot();
  s.t1 = next.t1;
  if (!s.move && next.move) { s.move = next.move; delete s.unresolved; }
  humanise(s);
  S.segs = S.segs.filter((x) => x.id !== next.id);
  changed();
}
function deleteSelected() {
  const s = selSeg(); if (!s) return;
  snapshot();
  const i = S.segs.indexOf(s);
  S.segs.splice(i, 1);
  S.sel = S.segs[i] ? S.segs[i].id : (S.segs[i - 1] ? S.segs[i - 1].id : null);
  changed();
}
function confirmSelected() {
  const s = selSeg(); if (!s) return;
  if (!s.move) { focusPicker(); return; }
  if (s.provenance !== 'human') { snapshot(); humanise(s); changed(); }
  const next = S.segs.find((x) => x.t0 >= s.t1 - 1e-6 && needsReview(x));
  if (next) { S.sel = next.id; seekT(next.t0 + 0.01); renderAll(); }
}
function confirmAll() {
  const n = S.segs.filter((s) => s.move && s.provenance === 'narration').length;
  if (!n) { toast('nothing narrated to confirm'); return; }
  if (!confirm(`Mark ${n} narrated segments as human-confirmed?`)) return;
  snapshot();
  for (const s of S.segs) if (s.move && s.provenance === 'narration') humanise(s);
  changed();
}
function selectAdjacent(d) {
  if (!S.segs.length) return;
  const s = selSeg();
  let i = s ? S.segs.indexOf(s) + d : (d > 0 ? 0 : S.segs.length - 1);
  if (!s) { const t = nowT(); i = d > 0 ? S.segs.findIndex((x) => x.t0 > t) : S.segs.map((x) => x.t0 < t).lastIndexOf(true); }
  i = clamp(i, 0, S.segs.length - 1);
  S.sel = S.segs[i].id; seekT(S.segs[i].t0 + 0.01); ensureVisible(S.segs[i].t0); renderAll();
}
function toggleLoop() {
  const s = selSeg();
  if (S.loop || !s) { S.loop = null; } else { S.loop = { t0: s.t0, t1: s.t1 }; seekT(s.t0); video.play(); }
  S.needDraw = true; toast(S.loop ? 'loop on (L to stop)' : 'loop off');
}
function newSegmentInGap(t) {
  const prev = S.segs.filter((s) => s.t1 <= t + 1e-6).sort((a, b) => b.t1 - a.t1)[0];
  const next = S.segs.filter((s) => s.t0 >= t - 1e-6).sort((a, b) => a.t0 - b.t0)[0];
  const halves = gridTimes('half');
  let t0 = halves.filter((g) => g <= t).pop(); if (t0 == null) t0 = t;
  if (prev && t0 < prev.t1) t0 = prev.t1;
  let t1 = halves.find((g) => g > t + 0.3); if (t1 == null) t1 = t + 2;
  if (next && t1 > next.t0) t1 = next.t0;
  if (t1 - t0 < MIN_LEN) return;
  snapshot();
  const s = { id: uid('seg'), t0, t1, move: null, node: null, tier: 'couple', provenance: 'human', confidence: null,
    annotator: S.info.annotator, transcript: null, notes: null, created: nowIso(), unresolved: '' };
  S.segs.push(s); S.sel = s.id; changed(); focusPicker();
}

// ------------------------------------------------------------------------------------------------ render: list + editor
function renderAll() { renderHeader(); renderList(); renderEditor(); S.needDraw = true; }
function renderHeader() {
  $('dirty').classList.toggle('hidden', !S.dirty);
  $('btnUndo').disabled = !S.undo.length;
  $('btnRedo').disabled = !S.redo.length;
}
function provBadge(s) {
  if (!s.move) return '<span class="badge unknown">unknown</span>';
  const low = s.confidence != null && s.confidence < 0.4 ? ' low' : '';
  return `<span class="badge ${esc(s.provenance)}${low}">${esc(s.provenance)}</span>`;
}
function renderList() {
  const list = $('segList');
  const rows = S.segs.map((s, i) => ({ s, i })).filter(({ s }) => S.filter === 'all' || needsReview(s));
  const nrev = S.segs.filter(needsReview).length;
  const secs = S.segs.filter((s) => s.move).reduce((a, s) => a + (s.t1 - s.t0), 0);
  $('listTitle').textContent = `Segments ${S.segs.length} (${secs.toFixed(1)} s labelled, ${nrev} to review)`;
  list.innerHTML = rows.map(({ s, i }) => {
    const name = s.move ? moveName(s.move) : `unknown: ${s.unresolved || '?'}`;
    return `<div class="seg${s.id === S.sel ? ' sel' : ''}" data-id="${esc(s.id)}">
      <span class="n">${i + 1}</span><span class="chip" style="background:${moveColor(s)}"></span>
      <span>${esc(name)}</span><span class="tt">${fmt(s.t0)} - ${fmt(s.t1)}</span>
      <span>${provBadge(s)}</span><span class="tt">${s.confidence == null ? '' : s.confidence.toFixed(2)}</span></div>`;
  }).join('') || '<div class="muted small" style="padding:6px">No segments yet: press R and narrate, or M / double-click the timeline to add one.</div>';
  const el = list.querySelector('.seg.sel');
  if (el) {  // scroll the list only (scrollIntoView would also scroll the panel column)
    const top = el.offsetTop, bot = top + el.offsetHeight;
    if (top < list.scrollTop) list.scrollTop = top;
    else if (bot > list.scrollTop + list.clientHeight) list.scrollTop = bot - list.clientHeight;
  }
}
function renderEditor() {
  const ed = $('editor');
  const s = selSeg();
  if (!s) {
    ed.innerHTML = '<div class="muted small">Select a segment (click it on the timeline or in the list). ' +
      '<kbd>?</kbd> shows the keys.</div>';
    return;
  }
  const m = S.moveBy[s.move];
  const match = s.match || {};
  const cands = (s.candidates || match.candidates || []).filter((c) => c.move && c.move !== s.move).slice(0, 4);
  const heard = s.unresolved || s.anchor_word || s.heard;
  let html = `<div class="edhead"><span class="chip" style="background:${moveColor(s)}"></span>
    <b>${esc(s.move ? moveName(s.move) : 'unknown: ' + (s.unresolved || '?'))}</b>
    ${m && m.name_en && m.name_en !== m.name ? `<span class="muted">(${esc(m.name_en)})</span>` : ''}
    ${m && m.status !== 'scaffold' ? `<span class="badge">${esc(m.status)}</span>` : ''}
    ${provBadge(s)} ${s.confidence != null ? `<span class="muted small">conf ${s.confidence.toFixed(2)}</span>` : ''}
    <span class="spacer"></span><span class="times small">
      <input id="edT0" type="number" step="0.01" value="${s.t0.toFixed(3)}"> -
      <input id="edT1" type="number" step="0.01" value="${s.t1.toFixed(3)}"> <span class="muted">${(s.t1 - s.t0).toFixed(2)} s</span></span></div>
    <div class="pickerWrap"><input id="picker" placeholder="change move: type a name, Portuguese name or alias (E)" autocomplete="off">
      <div id="pickList" class="hidden"></div></div>
    <div class="row">
      <button id="edConfirm" title="Enter">Confirm</button><button id="edSplit" title="S">Split at playhead</button>
      <button id="edMerge" title="J">Merge with next</button><button id="edDelete" title="Del">Delete</button>
      <button id="edLoop" class="${S.loop ? 'on' : ''}" title="L">Loop</button><button id="edPlay">Play from start</button></div>`;
  if (heard || cands.length || s.transcript) {
    html += `<div class="heard">`;
    if (heard) html += `heard <b>"${esc(heard)}"</b> `;
    if (match.method) html += `<span class="muted">(${esc([...new Set([match.status, match.method])].filter(Boolean).join(' '))} ${match.score != null ? match.score : ''})</span> `;
    if (s.relabelled_from) html += `<span class="muted">narration said ${esc(moveName(s.relabelled_from))}</span> `;
    if (s.transcript) html += `<div class="muted">"${esc(s.transcript)}"</div>`;
    if (cands.length) html += `<div>maybe: ${cands.map((c) => `<button data-cand="${esc(c.move)}">${esc(c.name || moveName(c.move))} <span class="muted">${c.score}</span></button>`).join('')}</div>`;
    if (s._teach && s.move) html += `<div>Teach the tool that "<b>${esc(s._teach)}</b>" means ${esc(moveName(s.move))}?
      <button id="edTeach" title="adds an alias in taxonomy/additions.toml">Remember alias</button></div>`;
    html += `</div>`;
  }
  if (!s.move) {
    const nm = (s.unresolved || '').replace(/\b\w/g, (c) => c.toUpperCase());
    html += `<div class="propose"><b>Not in the graph?</b> Create a proposed move (written to taxonomy/additions.toml):<br>
      <label>name <input id="pName" type="text" value="${esc(nm)}"></label>
      <label>family <select id="pFamily">${FAMILIES.map((f) => `<option${f === 'step' ? ' selected' : ''}>${f}</option>`).join('')}</select></label>
      <label>tone <select id="pTone">${Object.keys(TONES).map((t) => `<option>${t}</option>`).join('')}</select></label>
      <label>energy <input id="pEnergy" type="range" min="0" max="1" step="0.05" value="0.4"><span id="pEnergyV">0.40</span></label>
      <label>near <input id="pNear" type="text" placeholder="moves it connects to, comma separated"></label>
      <button id="pCreate">Create proposed move</button></div>`;
  }
  html += `<textarea id="edNotes" placeholder="notes">${esc(s.notes || '')}</textarea>`;
  ed.innerHTML = html;
  $('edConfirm').onclick = confirmSelected;
  $('edSplit').onclick = splitSelected;
  $('edMerge').onclick = mergeNext;
  $('edDelete').onclick = deleteSelected;
  $('edLoop').onclick = toggleLoop;
  $('edPlay').onclick = () => { seekT(s.t0); video.play(); };
  const setTime = (key, el) => {
    const v = parseFloat(el.value);
    if (!isFinite(v)) return;
    const other = S.segs.filter((x) => x.id !== s.id);
    let lo = key === 't0' ? -Infinity : s.t0 + MIN_LEN, hi = key === 't1' ? Infinity : s.t1 - MIN_LEN;
    for (const o of other) {
      if (key === 't0' && o.t1 <= s.t0 + 1e-6) lo = Math.max(lo, o.t1);
      if (key === 't1' && o.t0 >= s.t1 - 1e-6) hi = Math.min(hi, o.t0);
    }
    snapshot(); s[key] = clamp(v, lo, hi); humanise(s); changed();
  };
  $('edT0').onchange = (e) => setTime('t0', e.target);
  $('edT1').onchange = (e) => setTime('t1', e.target);
  $('edNotes').onchange = (e) => { snapshot(); s.notes = e.target.value || null; changed(); };
  ed.querySelectorAll('[data-cand]').forEach((b) => { b.onclick = () => assignMove(b.dataset.cand); });
  if ($('edTeach')) $('edTeach').onclick = () => teachAlias(s);
  if ($('pCreate')) {
    $('pEnergy').oninput = (e) => { $('pEnergyV').textContent = (+e.target.value).toFixed(2); };
    $('pCreate').onclick = () => proposeMove(s);
  }
  const pk = $('picker');
  pk.oninput = () => renderPick(pk.value);
  pk.onfocus = () => renderPick(pk.value);
  pk.onblur = () => setTimeout(() => $('pickList') && $('pickList').classList.add('hidden'), 150);
  pk.onkeydown = (e) => {
    if (e.key === 'ArrowDown') { S.pick.idx = Math.min(S.pick.idx + 1, S.pick.items.length - 1); renderPick(pk.value, true); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { S.pick.idx = Math.max(S.pick.idx - 1, 0); renderPick(pk.value, true); e.preventDefault(); }
    else if (e.key === 'Enter') { const m = S.pick.items[S.pick.idx]; if (m) { assignMove(m.slug); } e.preventDefault(); }
    else if (e.key === 'Escape') { pk.blur(); }
  };
}
function renderPick(q, keepIdx) {
  const pl = $('pickList');
  if (!pl) return;
  S.pick.items = searchMoves(q);
  if (!keepIdx) S.pick.idx = 0;
  pl.innerHTML = S.pick.items.map((m, i) => {
    const al = [m.name_en !== m.name ? m.name_en : null, ...(m.aliases || [])].filter(Boolean).slice(0, 5).join(', ');
    return `<div class="pick${i === S.pick.idx ? ' on' : ''}" data-slug="${esc(m.slug)}"><span class="chip" style="background:${TONES[m.tone] || '#4caf50'}"></span>
      <span>${esc(m.name)}</span>${m.status !== 'scaffold' ? `<span class="st">${esc(m.status)}</span>` : ''}<span class="al">${esc(al)}</span></div>`;
  }).join('') || '<div class="pick muted">no move matches</div>';
  pl.classList.remove('hidden');
  pl.querySelectorAll('[data-slug]').forEach((d) => { d.onmousedown = (e) => { e.preventDefault(); assignMove(d.dataset.slug); }; });
}
function focusPicker() { renderEditor(); const pk = $('picker'); if (pk) { pk.focus(); renderPick(''); } }

async function teachAlias(s) {
  try {
    const r = await api('POST', '/api/alias', { move: s.move, alias: s._teach });
    setMoves(r.moves); s._teach = null; toast('remembered: ' + r.message); renderEditor();
  } catch (e) { toast('alias not added: ' + e.message, true); }
}
async function proposeMove(s) {
  const name = $('pName').value.trim();
  if (!name) { toast('give the move a name', true); return; }
  const near = $('pNear').value.split(',').map((x) => x.trim()).filter(Boolean);
  const aliases = s.unresolved && norm(s.unresolved) !== norm(name) ? [s.unresolved] : [];
  if (!confirm(`Add the proposed move "${name}" to taxonomy/additions.toml and rebuild the graph?`)) return;
  try {
    const r = await api('POST', '/api/propose', { name, family: $('pFamily').value, tone: $('pTone').value,
      energy: parseFloat($('pEnergy').value), near, links_from: near.slice(0, 1), aliases });
    setMoves(r.moves); toast(r.message); assignMove(r.slug);
  } catch (e) { toast('not added: ' + e.message, true); }
}

// ------------------------------------------------------------------------------------------------ timeline drawing
const ctx2d = canvas.getContext('2d');
function xOf(t) { const v = S.view; return (t - v.t0) / (v.t1 - v.t0) * canvas.clientWidth; }
function tOf(x) { const v = S.view; return v.t0 + x / canvas.clientWidth * (v.t1 - v.t0); }
function ensureVisible(t) {
  const v = S.view;
  if (!v || !isFinite(t)) return;
  const w = v.t1 - v.t0;
  if (t < v.t0 || t > v.t1) { S.view = { t0: t - 0.1 * w, t1: t + 0.9 * w }; S.needDraw = true; }
}
function zoom(f, tc) {
  const v = S.view; tc = tc == null ? (v.t0 + v.t1) / 2 : tc;
  const w = clamp((v.t1 - v.t0) * f, 1, Math.max(10, (S.range[1] - S.range[0]) * 1.2));
  const a = (tc - v.t0) / (v.t1 - v.t0);
  S.view = { t0: tc - a * w, t1: tc - a * w + w }; S.needDraw = true;
}
function hatch(x, y, w, h, color) {
  ctx2d.save(); ctx2d.beginPath(); ctx2d.rect(x, y, w, h); ctx2d.clip();
  ctx2d.strokeStyle = color; ctx2d.lineWidth = 1;
  for (let k = -h; k < w; k += 7) { ctx2d.beginPath(); ctx2d.moveTo(x + k, y + h); ctx2d.lineTo(x + k + h, y); ctx2d.stroke(); }
  ctx2d.restore();
}
function draw() {
  S.needDraw = false;
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth, H = canvas.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
  const c = ctx2d;
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, W, H);
  c.fillStyle = '#16181e'; c.fillRect(0, 0, W, H);
  if (!S.view) return;
  const v = S.view, pxs = W / (v.t1 - v.t0);
  // available video range
  c.fillStyle = '#1b1e25';
  c.fillRect(xOf(S.range[0]), 0, xOf(S.range[1]) - xOf(S.range[0]), H);
  // narrated spans
  c.fillStyle = 'rgba(90,169,255,0.10)';
  for (const sp of S.spans) c.fillRect(xOf(sp[0]), TL.words[0], xOf(sp[1]) - xOf(sp[0]), TL.words[1] - TL.words[0]);
  // proposals coverage
  if (S.proposals && S.proposals.coverage) {
    c.fillStyle = 'rgba(200,160,255,0.05)';
    c.fillRect(xOf(S.proposals.coverage[0]), TL.segs[0], xOf(S.proposals.coverage[1]) - xOf(S.proposals.coverage[0]), TL.segs[1] - TL.segs[0]);
  }
  // beat grid
  const eighthPx = S.grid.length > 1 ? (S.grid[1][0] - S.grid[0][0]) * pxs : 0;
  let mi = 0;
  c.font = '10px system-ui'; c.textBaseline = 'top';
  for (const g of S.grid) {
    if (g[2] === 0) mi++;
    if (g[0] < v.t0 - 1 || g[0] > v.t1 + 1) continue;
    const x = Math.round(xOf(g[0])) + 0.5;
    if (g[2] === 0) {
      c.strokeStyle = 'rgba(255,255,255,0.22)'; c.beginPath(); c.moveTo(x, TL.beats[0]); c.lineTo(x, TL.words[1]); c.stroke();
      if (eighthPx * 8 > 26) { c.fillStyle = '#9aa1b2'; c.fillText(String(mi), x + 2, TL.beats[0] + 2); }
    } else if (g[2] === 4) {
      c.strokeStyle = 'rgba(255,255,255,0.11)'; c.beginPath(); c.moveTo(x, TL.beats[0] + 6); c.lineTo(x, TL.segs[1]); c.stroke();
    } else if (eighthPx > 5) {
      c.strokeStyle = 'rgba(255,255,255,0.12)'; c.beginPath(); c.moveTo(x, TL.beats[1] - 5); c.lineTo(x, TL.beats[1]); c.stroke();
    }
  }
  // ruler
  const steps = [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
  const st = steps.find((s) => s * pxs >= 70) || 600;
  c.fillStyle = '#7d8495'; c.strokeStyle = '#3a3f4d';
  for (let t = Math.ceil(v.t0 / st) * st; t <= v.t1; t += st) {
    const x = Math.round(xOf(t)) + 0.5;
    c.beginPath(); c.moveTo(x, 10); c.lineTo(x, 16); c.stroke();
    c.fillText(fmt(t).replace(/\.00$/, ''), x + 2, 1);
  }
  // counterbalance
  for (const iv of S.cb) {
    c.fillStyle = (iv.confidence != null && iv.confidence < 0.4) ? 'rgba(255,226,59,0.35)' : 'rgba(255,226,59,0.85)';
    c.fillRect(xOf(iv.t0), TL.cb[0], xOf(iv.t1) - xOf(iv.t0), TL.cb[1] - TL.cb[0]);
  }
  // proposal boundaries
  if (S.proposals) {
    c.setLineDash([3, 4]);
    for (const b of S.proposals.boundaries || []) {
      const x = Math.round(xOf(b.t)) + 0.5;
      c.strokeStyle = `rgba(200,160,255,${0.25 + 0.6 * clamp(b.strength || 0, 0, 1)})`;
      c.beginPath(); c.moveTo(x, TL.cb[0]); c.lineTo(x, TL.segs[1] + 2); c.stroke();
    }
    c.setLineDash([]);
  }
  // segments
  const [sy0, sy1] = TL.segs, sh = sy1 - sy0;
  c.font = '12px system-ui'; c.textBaseline = 'middle';
  for (const s of S.segs) {
    if (s.t1 < v.t0 || s.t0 > v.t1) continue;
    const x0 = xOf(s.t0), x1 = xOf(s.t1), w = Math.max(1, x1 - x0);
    const col = moveColor(s);
    c.globalAlpha = s.provenance === 'human' ? 0.85 : (s.provenance === 'placeholder' ? 0.25 : 0.5);
    c.fillStyle = col; c.fillRect(x0, sy0, w, sh);
    c.globalAlpha = 1;
    if (!s.move) hatch(x0, sy0, w, sh, 'rgba(255,255,255,0.35)');
    else if (s.provenance === 'narration') hatch(x0, sy0, w, sh, 'rgba(0,0,0,0.35)');
    if (s.provenance === 'auto') { c.setLineDash([4, 3]); }
    c.lineWidth = s.id === S.sel ? 2 : 1;
    c.strokeStyle = s.id === S.sel ? '#ffffff' : ((s.confidence != null && s.confidence < 0.4) ? '#ffb347' : 'rgba(0,0,0,0.6)');
    c.strokeRect(x0 + 0.5, sy0 + 0.5, w - 1, sh - 1);
    c.setLineDash([]);
    if (w > 18) {
      c.save(); c.beginPath(); c.rect(x0 + 3, sy0, w - 6, sh); c.clip();
      const dark = s.move && S.moveBy[s.move] && S.moveBy[s.move].tone === 'yellow';
      c.fillStyle = dark ? '#111' : '#fff';
      c.fillText(s.move ? moveName(s.move) : `? ${s.unresolved || ''}`, x0 + 4, sy0 + sh / 2 - 7);
      c.font = '10px system-ui';
      c.fillStyle = dark ? '#333' : 'rgba(255,255,255,0.75)';
      c.fillText(`${s.provenance}${s.confidence != null ? ' ' + s.confidence.toFixed(2) : ''}`, x0 + 4, sy0 + sh / 2 + 9);
      c.font = '12px system-ui';
      c.restore();
    }
  }
  // loop
  if (S.loop) { c.fillStyle = 'rgba(89,211,140,0.12)'; c.fillRect(xOf(S.loop.t0), 0, xOf(S.loop.t1) - xOf(S.loop.t0), H); }
  // narration words + taps
  c.font = '10px system-ui'; c.textBaseline = 'top';
  let lastX = -1e9;
  for (const w of S.words) {
    if (w.t < v.t0 || w.t > v.t1) continue;
    const x = xOf(w.t);
    c.strokeStyle = w.hit ? '#5aa9ff' : 'rgba(255,255,255,0.3)';
    c.beginPath(); c.moveTo(x + 0.5, TL.words[0]); c.lineTo(x + 0.5, TL.words[0] + 8); c.stroke();
    const tw = c.measureText(w.word).width;
    const row = x < lastX + 4 ? 1 : 0;
    if (row === 0) lastX = x + tw;
    c.fillStyle = w.hit ? '#9fd0ff' : '#8d93a3';
    c.fillText(w.word, x + 2, TL.words[0] + 10 + 14 * row);
  }
  c.fillStyle = '#ffb347';
  for (const t of S.taps) {
    const x = xOf(t);
    c.beginPath(); c.moveTo(x, TL.words[1] - 8); c.lineTo(x - 4, TL.words[1]); c.lineTo(x + 4, TL.words[1]); c.fill();
  }
  // drag preview
  if (S.drag && S.drag.kind === 'edge' && S.drag.t != null) {
    const x = xOf(S.drag.t);
    c.strokeStyle = '#fff'; c.setLineDash([2, 2]); c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); c.setLineDash([]);
    c.fillStyle = '#fff'; c.fillText(fmt(S.drag.t), x + 3, 2);
  }
  // playhead
  if (S.cam) {
    const x = Math.round(xOf(nowT())) + 0.5;
    c.strokeStyle = S.rec ? '#ff4d5e' : '#ff6b6b'; c.lineWidth = 1.5;
    c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); c.lineWidth = 1;
  }
}

// ------------------------------------------------------------------------------------------------ timeline mouse
function hitSegment(x, y) {
  if (y < TL.segs[0] || y > TL.segs[1]) return null;
  const edges = [];
  for (const s of S.segs) {
    for (const side of ['t0', 't1']) {
      const d = Math.abs(xOf(s[side]) - x);
      if (d <= 6) edges.push({ s, side, d: d - (s.id === S.sel ? 2 : 0) });
    }
  }
  if (edges.length) { edges.sort((a, b) => a.d - b.d); return { kind: 'edge', ...edges[0] }; }
  const t = tOf(x);
  const s = S.segs.find((q) => q.t0 <= t && t < q.t1);
  return s ? { kind: 'body', s } : null;
}
canvas.addEventListener('mousedown', (e) => {
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const h = hitSegment(x, y);
  if (h && h.kind === 'edge') {
    const t = h.s[h.side];
    const linked = S.segs.filter((q) => q !== h.s && Math.abs(q[h.side === 't0' ? 't1' : 't0'] - t) < 1e-3)
      .map((q) => ({ s: q, side: h.side === 't0' ? 't1' : 't0' }));
    S.drag = { kind: 'edge', items: [{ s: h.s, side: h.side }, ...(e.ctrlKey ? [] : linked)], t0: t, t: null, moved: false };
    S.sel = h.s.id; renderList(); renderEditor();
  } else if (h && h.kind === 'body') {
    S.sel = h.s.id; renderList(); renderEditor(); S.needDraw = true;
    S.drag = { kind: 'pan', x0: x, view0: { ...S.view }, moved: false, seekOnClick: true };
  } else {
    S.drag = { kind: 'pan', x0: x, view0: { ...S.view }, moved: false, seekOnClick: true };
  }
});
window.addEventListener('mousemove', (e) => {
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  const d = S.drag;
  if (!d) {
    if (x >= 0 && y >= 0 && x <= r.width && y <= r.height) {
      const h = hitSegment(x, y);
      canvas.style.cursor = h && h.kind === 'edge' ? 'ew-resize' : 'crosshair';
      const t = tOf(x);
      const seg = h && h.s;
      $('tlInfo').innerHTML = `${fmt(t)} ${measureOf(t)}` + (seg ? ` &nbsp; <b>${esc(seg.move ? moveName(seg.move) : 'unknown: ' + (seg.unresolved || '?'))}</b> ${fmt(seg.t0)} - ${fmt(seg.t1)} ${esc(seg.provenance)}${seg.anchor_word ? ` heard "${esc(seg.anchor_word)}"` : ''}` : '');
    }
    return;
  }
  if (d.kind === 'pan') {
    if (Math.abs(x - d.x0) > 3) d.moved = true;
    if (d.moved) {
      const dt = (x - d.x0) / canvas.clientWidth * (d.view0.t1 - d.view0.t0);
      S.view = { t0: d.view0.t0 - dt, t1: d.view0.t1 - dt }; S.needDraw = true;
    }
  } else if (d.kind === 'edge') {
    let t = tOf(x);
    if (!e.altKey) {  // magnetic: the playhead and other edges within 8 px; otherwise the nearest grid eighth
      let best = null; const tolT = 8 / canvas.clientWidth * (S.view.t1 - S.view.t0);
      const cands = [nowT()];
      for (const s of S.segs) if (!d.items.some((it) => it.s === s)) cands.push(s.t0, s.t1);
      for (const ct of cands) { const dd = Math.abs(ct - t); if (dd <= tolT && (best == null || dd < Math.abs(best - t))) best = ct; }
      if (best == null && S.grid.length) {
        for (const g of S.grid) if (best == null || Math.abs(g[0] - t) < Math.abs(best - t)) best = g[0];
      }
      if (best != null) t = best;
    }
    let lo = -Infinity, hi = Infinity;
    for (const it of d.items) {
      if (it.side === 't0') hi = Math.min(hi, it.s.t1 - MIN_LEN); else lo = Math.max(lo, it.s.t0 + MIN_LEN);
      for (const o of S.segs) {
        if (d.items.some((q) => q.s === o)) continue;
        if (it.side === 't0' && o.t1 <= it.s.t0 + 1e-6) lo = Math.max(lo, o.t1);
        if (it.side === 't1' && o.t0 >= it.s.t1 - 1e-6) hi = Math.min(hi, o.t0);
      }
    }
    t = clamp(t, lo, hi);
    if (!d.moved && Math.abs(t - d.t0) > 1e-4) { snapshot(); d.moved = true; }
    if (d.moved) { for (const it of d.items) it.s[it.side] = t; d.t = t; S.needDraw = true; }
  }
});
window.addEventListener('mouseup', (e) => {
  const d = S.drag; S.drag = null;
  if (!d) return;
  if (d.kind === 'edge' && d.moved) { for (const it of d.items) humanise(it.s); changed(); }
  else if (d.kind === 'pan' && !d.moved && d.seekOnClick) {
    const r = canvas.getBoundingClientRect();
    if (e.clientX >= r.left && e.clientX <= r.right) seekT(tOf(e.clientX - r.left));
  }
  S.needDraw = true;
});
canvas.addEventListener('dblclick', (e) => {
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
  if (y >= TL.segs[0] && y <= TL.segs[1] && !hitSegment(x, y)) newSegmentInGap(tOf(x));
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = canvas.getBoundingClientRect(), x = e.clientX - r.left;
  if (e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) {
    const dt = (e.shiftKey ? e.deltaY : e.deltaX) / canvas.clientWidth * (S.view.t1 - S.view.t0);
    S.view = { t0: S.view.t0 + dt, t1: S.view.t1 + dt }; S.needDraw = true;
  } else zoom(Math.pow(1.0015, e.deltaY), tOf(x));
}, { passive: false });
window.addEventListener('resize', () => { S.needDraw = true; });

// ------------------------------------------------------------------------------------------------ transport
function setRate(r) {
  S.settings.speed = r; store('speed', r);
  video.playbackRate = r;
  document.querySelectorAll('.speeds button').forEach((b) => b.classList.toggle('on', +b.dataset.rate === r));
}
function togglePlay() { if (video.paused) video.play(); else video.pause(); }
function step(dir, measure) {
  const t = nowT();
  const pts = S.grid.length ? gridTimes(measure ? 'measure' : 'eighth') : null;
  let nt;
  if (pts && pts.length) {
    nt = dir > 0 ? pts.find((g) => g > t + 0.02) : pts.filter((g) => g < t - 0.02).pop();
    if (nt == null) nt = t + dir * (measure ? 1 : 0.1);
  } else nt = t + dir * (measure ? 1 : 0.1);
  video.pause(); seekT(nt); ensureVisible(nt);
}
video.addEventListener('play', () => { $('btnPlay').textContent = 'Pause'; recLog('play'); });
video.addEventListener('pause', () => { $('btnPlay').textContent = 'Play'; recLog('pause'); });
video.addEventListener('seeking', () => { if (!S.switching) recLog('seeking'); S.needDraw = true; });
video.addEventListener('seeked', () => { if (!S.switching) recLog('seeked'); S.needDraw = true; });
video.addEventListener('ratechange', () => recLog('ratechange'));
video.addEventListener('error', () => cameraProblem('the browser could not load this video'));

function cameraProblem(msg) {
  const c = camObj();
  const n = $('videoNote');
  n.innerHTML = `Camera ${esc(c.id)} (${esc(c.codec || '?')}): ${esc(msg)}. Pick another camera` +
    (c.codec && c.codec !== 'h264' ? `, or <button id="btnProxy">make an H.264 proxy</button> (ffmpeg on the CPU: about 20 s per minute of 1080p video; frame times are kept exactly)` : '') + '.';
  n.classList.remove('hidden');
  if ($('btnProxy')) $('btnProxy').onclick = makeProxy;
}
async function makeProxy() {
  const id = S.cam;
  $('videoNote').textContent = 'making a proxy with ffmpeg ...';
  try {
    const { job } = await api('POST', '/api/proxy/' + id);
    const j = await pollJob(job);
    if (j.state !== 'done') throw new Error(j.error || 'failed');
    S.cams[id].proxy = true; S.cams[id].bust = Date.now();
    loadCamera(id);
    toast('proxy ready');
  } catch (e) { cameraProblem('proxy failed: ' + e.message); }
}
function loadCamera(id) {
  const keep = S.cam && video.readyState > 0 ? nowT() : null;
  const wasPlaying = !video.paused;
  S.cam = id; S.switching = true;
  $('camSel').value = id;
  $('videoNote').classList.add('hidden');
  const c = camObj();
  video.src = c.url + (c.bust ? '?v=' + c.bust : '');
  video.load();
  video.addEventListener('loadedmetadata', () => {
    video.playbackRate = S.settings.speed;
    S.range = [toLabel(0), toLabel(video.duration)];
    if (!S.view) {
      const segs = S.segs.length ? [S.segs[0].t0, S.segs[S.segs.length - 1].t1] : null;
      S.view = segs ? { t0: segs[0] - 5, t1: segs[1] + 5 } : { t0: S.range[0], t1: S.range[1] };
      if (S.view.t1 - S.view.t0 < 10) zoom(10 / (S.view.t1 - S.view.t0));
    }
    if (video.videoWidth === 0) cameraProblem('this browser decodes no picture from it');
    const done = () => { S.switching = false; if (S.rec) recLog('camera'); if (wasPlaying) video.play(); S.needDraw = true; };
    if (keep != null) {
      video.addEventListener('seeked', done, { once: true });
      video.currentTime = clamp(toVideo(keep), 0, video.duration - 0.05);
    } else done();
  }, { once: true });
}

function frame() {
  if (S.cam && video.readyState > 0) {
    const t = nowT();
    if (S.loop && !video.paused && t >= S.loop.t1) seekT(S.loop.t0);
    if (!video.paused) {
      S.needDraw = true;
      const v = S.view;
      if (v && !S.drag && (t > v.t1 - 0.02 * (v.t1 - v.t0) || t < v.t0)) { const w = v.t1 - v.t0; S.view = { t0: t - 0.05 * w, t1: t + 0.95 * w }; }
    }
    $('clock').innerHTML = `${fmt(t)} <span>${measureOf(t)}</span> &nbsp; camera ${esc(S.cam)} ${fmt(video.currentTime)}` +
      (S.info && S.info.mode === 'take' ? ' (reference time)' : '');
  }
  if (S.rec) recMeter();
  if (S.needDraw) draw();
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------------------------------------ narration recording
function recLog(ev) {
  const R = S.rec;
  if (!R || (S.switching && ev !== 'camera')) return;
  const e = { v: +video.currentTime.toFixed(4), cam: S.cam, rate: video.playbackRate, paused: video.paused, ev };
  if (R.kind === 'worklet') e.c = R.ctx.currentTime; else e.a = performance.now() / 1000 - R.t0;
  R.log.push(e);
}
function recMeter() {
  const R = S.rec;
  if (!R || !R.analyser) return;
  const buf = new Float32Array(R.analyser.fftSize);
  R.analyser.getFloatTimeDomainData(buf);
  let s = 0; for (const x of buf) s += x * x;
  const rms = Math.sqrt(s / buf.length), lc = $('level').getContext('2d');
  lc.fillStyle = '#111'; lc.fillRect(0, 0, 120, 10);
  lc.fillStyle = rms > 0.3 ? '#ff4d5e' : '#59d38c'; lc.fillRect(0, 0, Math.min(120, rms * 400), 10);
  const secs = R.kind === 'worklet' ? R.ctx.currentTime - R.ctxStart : performance.now() / 1000 - R.t0;
  $('recStatus').textContent = `recording ${fmt(secs)} - say the move names; M = mark; R = stop`;
}
async function toggleRec() { if (S.rec) await stopRec(); else await startRec(); }
async function startRec() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { toast('no microphone access in this browser', true); return; }
  let stream;
  try { stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } }); }
  catch (e) { toast('microphone refused: ' + e.message, true); return; }
  const R = { stream, log: [], started: nowIso(), cam0: S.cam };
  S.taps = [];
  try {
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule('/static/pcm-worklet.js');
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'pcm-recorder', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
    const mute = ctx.createGain(); mute.gain.value = 0;
    src.connect(node); node.connect(mute); mute.connect(ctx.destination);
    const analyser = ctx.createAnalyser(); analyser.fftSize = 1024; src.connect(analyser);
    Object.assign(R, { kind: 'worklet', ctx, node, analyser, sr: ctx.sampleRate, ctxStart: ctx.currentTime,
      firstFrame: null, out: [], acc: 0, accN: 0, pos: 0, cur: new Int16Array(16000), curN: 0 });
    node.port.onmessage = (ev) => {
      if (ev.data.done) { R.finished && R.finished(); return; }
      if (R.firstFrame == null) R.firstFrame = ev.data.frame;
      decimate(R, ev.data.data);
    };
  } catch (e) {
    // fallback: MediaRecorder (clock = performance.now() from its start event)
    const mr = new MediaRecorder(stream);
    Object.assign(R, { kind: 'mediarecorder', mr, chunks: [], t0: performance.now() / 1000 });
    mr.ondataavailable = (ev) => { if (ev.data && ev.data.size) R.chunks.push(ev.data); };
    mr.onstart = () => { R.t0 = performance.now() / 1000; };
    mr.start(1000);
  }
  S.rec = R;
  recLog('start');
  R.timer = setInterval(() => recLog('tick'), 100);
  $('btnRec').classList.add('recording'); $('btnRec').textContent = '■ Stop narrating';
  if (video.paused) video.play();
  toast(R.kind === 'worklet' ? 'narrating: say the move names' : 'narrating (MediaRecorder fallback)');
}
function decimate(R, x) {
  // streaming box-filter decimation to 16 kHz int16 (any input rate)
  const k = 16000 / R.sr;
  for (let i = 0; i < x.length; i++) {
    R.acc += x[i]; R.accN++; R.pos += k;
    if (R.pos >= 1) {
      R.pos -= 1;
      const v = clamp(R.acc / R.accN, -1, 1);
      R.cur[R.curN++] = v < 0 ? v * 32768 : v * 32767;
      R.acc = 0; R.accN = 0;
      if (R.curN === R.cur.length) { R.out.push(R.cur); R.cur = new Int16Array(16000); R.curN = 0; }
    }
  }
}
function wavBlob(chunks, sr) {
  const n = chunks.reduce((a, c) => a + c.length, 0);
  const buf = new ArrayBuffer(44 + 2 * n), dv = new DataView(buf);
  const wr = (o, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(o + i, s.charCodeAt(i)); };
  wr(0, 'RIFF'); dv.setUint32(4, 36 + 2 * n, true); wr(8, 'WAVE'); wr(12, 'fmt ');
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true); dv.setUint32(24, sr, true);
  dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true); wr(36, 'data'); dv.setUint32(40, 2 * n, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++) { dv.setInt16(o, c[i], true); o += 2; }
  return new Blob([buf], { type: 'audio/wav' });
}
async function stopRec() {
  const R = S.rec;
  if (!R) return;
  recLog('stop');
  clearInterval(R.timer);
  video.pause();
  $('btnRec').classList.remove('recording'); $('btnRec').innerHTML = '&#9679; Narrate';
  let blob, logDoc;
  if (R.kind === 'worklet') {
    await new Promise((res) => { R.finished = res; R.node.port.postMessage('stop'); setTimeout(res, 1500); });
    R.stream.getTracks().forEach((t) => t.stop());
    R.out.push(R.cur.slice(0, R.curN));
    const base = R.firstFrame != null ? R.firstFrame / R.sr : R.ctxStart;
    for (const e of R.log) { e.a = +(e.c - base).toFixed(4); delete e.c; }
    blob = wavBlob(R.out, 16000);
    logDoc = { clock: 'audioworklet', context_sample_rate: R.sr, sample_rate: 16000, base_latency: R.ctx.baseLatency,
      entries: R.log };
    R.ctx.close();
  } else {
    await new Promise((res) => { R.mr.onstop = res; R.mr.stop(); });
    R.stream.getTracks().forEach((t) => t.stop());
    blob = new Blob(R.chunks, { type: R.mr.mimeType || 'audio/webm' });
    logDoc = { clock: 'mediarecorder', entries: R.log };
  }
  S.rec = null;
  Object.assign(logDoc, { started: R.started, camera: R.cam0, time_base: S.info.time_base, take: S.info.name,
    user_agent: navigator.userAgent });
  $('recStatus').textContent = `uploading ${(blob.size / 1e6).toFixed(1)} MB ...`;
  const fd = new FormData();
  fd.append('audio', blob, R.kind === 'worklet' ? 'audio.wav' : 'audio.webm');
  fd.append('log', JSON.stringify(logDoc));
  fd.append('settings', JSON.stringify(currentSettings()));
  try {
    const { id, job } = await api('POST', '/api/narration', fd, true);
    $('recStatus').textContent = 'transcribing locally ...';
    const j = await pollJob(job, (jj) => { if (jj.log && jj.log.length) $('recStatus').textContent = jj.log[jj.log.length - 1]; });
    if (j.state !== 'done') throw new Error(j.error || 'failed');
    showNarration(j.result);
    $('recStatus').textContent = '';
    refreshNarrationList(id);
  } catch (e) { $('recStatus').textContent = ''; toast('narration failed: ' + e.message, true); }
}
function currentSettings() {
  return { lag: parseFloat($('lag').value) || 0, snap: $('snap').value, engine: $('engine').value,
    typesafe: $('typesafe').checked, annotator: S.info.annotator };
}

// ------------------------------------------------------------------------------------------------ narration results
function showNarration(res) {
  S.narr = res;
  S.words =(res.words || []).map((w) => ({ word: w.word, t: w.t, hit: (res.events || []).some((e) => (e.kind === 'move' || e.kind === 'unknown') && w.a >= e.a0 - 1e-3 && w.a <= e.a1 + 1e-3) }));
  S.spans = res.spans || [];
  S.taps = res.taps || [];
  $('narrResult').classList.remove('hidden');
  const sm = res.summary || {};
  $('narrTitle').textContent = `${res.id}: ${sm.moves || 0} moves, ${sm.unknown || 0} unknown (${res.engine || 'no recogniser'})`;
  $('narrWarn').textContent = (res.warnings || []).join(' | ');
  const evs = (res.events || []).filter((e) => e.kind !== 'ignored' || e.phrase);
  $('narrEvents').innerHTML = evs.map((e, i) => {
    const t = e.t0 != null ? e.t0 : e.anchor_t;
    const what = e.kind === 'move' ? `${esc(e.name || e.move)} <span class="muted">${esc(e.status || '')} ${e.confidence != null ? e.confidence.toFixed(2) : ''}</span>`
      : e.kind === 'unknown' ? `unknown${(e.candidates || []).length ? ` <span class="muted">(${esc(e.candidates.slice(0, 2).map((c) => c.name || c.move || c.term).join(' / '))}?)</span>` : ''}`
      : e.kind === 'control' ? `${esc(e.control)}` : e.kind;
    return `<div class="ev ${esc(e.kind)}" data-t="${t}"><span class="t">${fmt(t)}</span><span>"${esc(e.phrase || '')}"</span><span class="m">${what}</span><span class="t">${esc(e.snap || '')}</span></div>`;
  }).join('') || '<div class="muted small">nothing recognised</div>';
  $('narrEvents').querySelectorAll('.ev').forEach((d) => { d.onclick = () => { const t = +d.dataset.t; if (isFinite(t)) { seekT(t); ensureVisible(t); } }; });
  if (S.spans.length) ensureVisible(S.spans[0][0]);
  S.needDraw = true;
}
async function applyNarration() {
  if (!S.narr) return;
  const r = await api('POST', '/api/merge', { current: S.segs, new: S.narr.segments || [] });
  snapshot();
  S.segs = r.segments;
  const first = (S.narr.segments || [])[0];
  if (first) S.sel = first.id;
  changed();
  toast(`applied ${(S.narr.segments || []).length} narrated segments - review them (Enter confirms)`);
}
async function remapNarration() {
  if (!S.narr) return;
  try {
    const r = await api('POST', `/api/narration/${S.narr.id}/remap`, { settings: currentSettings() });
    if (r.job) { const j = await pollJob(r.job); if (j.state !== 'done') throw new Error(j.error); showNarration(j.result); }
    else showNarration(r);
    toast('re-mapped');
  } catch (e) { toast('re-map failed: ' + e.message, true); }
}
async function importTranscript(file) {
  if (!S.narr || !file) return;
  try {
    const words = JSON.parse(await file.text());
    const r = await fetch(`/api/narration/${S.narr.id}/transcript`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(words) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    showNarration(d); toast('transcript imported');
  } catch (e) { toast('transcript import failed: ' + e.message, true); }
}
async function openNarration(id) {
  if (!id) return;
  try { showNarration(await api('GET', '/api/narration/' + id)); } catch (e) { toast('cannot open: ' + e.message, true); }
}
async function refreshNarrationList(select) {
  try {
    const list = await api('GET', '/api/narrations');
    $('narrSel').innerHTML = list.map((n) => `<option value="${esc(n.id)}">${esc(n.id)}${n.summary ? ` (${n.summary.moves} moves)` : n.has_transcript ? '' : ' (not processed)'}</option>`).join('');
    if (select) $('narrSel').value = select;
  } catch (e) { /* ignore */ }
}

// ------------------------------------------------------------------------------------------------ save / export
async function save() {
  try {
    const r = await api('POST', '/api/labels', { segments: S.segs });
    S.dirty = false; renderHeader();
    $('savedInfo').textContent = `saved ${new Date().toLocaleTimeString()}: ${r.segments} segments` + (r.unresolved ? ` + ${r.unresolved} unknown` : '');
    toast(`labels.json saved (${r.segments} segments, ${r.labelled_seconds} s, ${r.provenance})`);
    return true;
  } catch (e) {
    toast('NOT saved:\n' + e.message, true);
    return false;
  }
}
async function exportUnity() {
  if (S.info.mode !== 'take') { toast('export needs a take (this is a lone video)', true); return; }
  if (S.dirty && !(await save())) return;
  $('btnExport').disabled = true;
  try {
    const { job } = await api('POST', '/api/export');
    toast('exporting: path + fingerprint + Unity copies ...');
    const j = await pollJob(job);
    const r = j.result || {};
    showModal(j.state === 'done' && r.returncode === 0 ? 'Export finished' : 'Export problem',
      (j.error || '') + (r.stdout || '') + (r.stderr ? '\n' + r.stderr : ''));
  } catch (e) { toast('export failed: ' + e.message, true); }
  $('btnExport').disabled = false;
}
function showModal(title, text) { $('modalTitle').textContent = title; $('modalText').textContent = text; $('modal').classList.remove('hidden'); }

// ------------------------------------------------------------------------------------------------ keyboard
document.addEventListener('keydown', (e) => {
  const tag = (e.target.tagName || '').toLowerCase();
  const typing = tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;
  const k = e.key;
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 's') { e.preventDefault(); save(); return; }
  if (typing) { if (k === 'Escape') e.target.blur(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); return; }
  if ((e.ctrlKey || e.metaKey) && k.toLowerCase() === 'y') { e.preventDefault(); redo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (k) {
    case ' ': e.preventDefault(); togglePlay(); break;
    case 'ArrowLeft': e.preventDefault(); step(-1, e.shiftKey); break;
    case 'ArrowRight': e.preventDefault(); step(1, e.shiftKey); break;
    case 'm': case 'M': mark(); break;
    case 's': case 'S': splitSelected(); break;
    case 'j': case 'J': mergeNext(); break;
    case 'Delete': case 'Backspace': e.preventDefault(); deleteSelected(); break;
    case 'Enter': e.preventDefault(); confirmSelected(); break;
    case 'e': case 'E': e.preventDefault(); focusPicker(); break;
    case 'l': case 'L': toggleLoop(); break;
    case 'r': case 'R': toggleRec(); break;
    case '[': selectAdjacent(-1); break;
    case ']': selectAdjacent(1); break;
    case '1': setRate(0.25); break;
    case '2': setRate(0.5); break;
    case '3': setRate(0.75); break;
    case '4': setRate(1); break;
    case 'Home': seekT(S.range[0]); ensureVisible(S.range[0]); break;
    case 'End': seekT(S.range[1]); ensureVisible(S.range[1]); break;
    case '+': case '=': zoom(1 / 1.5, nowT()); break;
    case '-': case '_': zoom(1.5, nowT()); break;
    case '?': $('help').classList.toggle('hidden'); break;
    case 'Escape': $('help').classList.add('hidden'); $('modal').classList.add('hidden'); S.sel = null; S.loop = null; renderAll(); break;
    default: break;
  }
});

// ------------------------------------------------------------------------------------------------ boot
function wire() {
  $('btnSave').onclick = save;
  $('btnUndo').onclick = undo;
  $('btnRedo').onclick = redo;
  $('btnExport').onclick = exportUnity;
  $('btnHelp').onclick = () => $('help').classList.toggle('hidden');
  $('helpClose').onclick = () => $('help').classList.add('hidden');
  $('modalClose').onclick = () => $('modal').classList.add('hidden');
  $('btnPlay').onclick = togglePlay;
  $('btnRec').onclick = toggleRec;
  $('camSel').onchange = (e) => loadCamera(e.target.value);
  document.querySelectorAll('.speeds button').forEach((b) => { b.onclick = () => setRate(+b.dataset.rate); });
  $('btnApply').onclick = applyNarration;
  $('btnRemap').onclick = remapNarration;
  $('btnDiscard').onclick = () => { S.narr = null; S.words = []; S.spans = []; S.taps = []; $('narrResult').classList.add('hidden'); S.needDraw = true; };
  $('btnNarrAudio').onclick = () => { if (!S.narr) return; const a = $('narrAudio'); a.src = `/api/narration/${S.narr.id}/audio`; a.play(); };
  $('trFile').onchange = (e) => importTranscript(e.target.files[0]);
  $('btnOpenNarr').onclick = () => openNarration($('narrSel').value);
  $('listFilter').onchange = (e) => { S.filter = e.target.value; renderList(); };
  $('btnConfirmAll').onclick = confirmAll;
  $('segList').onclick = (e) => {
    const row = e.target.closest('.seg'); if (!row) return;
    S.sel = row.dataset.id; const s = selSeg(); if (s) { seekT(s.t0 + 0.01); ensureVisible(s.t0); }
    renderAll();
  };
  for (const id of ['lag', 'snap', 'engine', 'typesafe']) {
    $(id).onchange = () => { const st = currentSettings(); store('settings', { lag: st.lag, snap: st.snap, engine: st.engine, typesafe: st.typesafe }); };
  }
  window.addEventListener('beforeunload', (e) => { if (S.dirty || S.rec) { e.preventDefault(); e.returnValue = ''; } });
}
function banner(html) { const b = $('banner'); b.innerHTML = html; b.classList.toggle('hidden', !html); }
function asrText(a) {
  if (!a) return 'checking speech recognisers ...';
  const fw = a['faster-whisper'] || {}, sp = a.sapi || {};
  if (fw.available) return `speech: faster-whisper ${fw.model || ''} (${(fw.cuda_devices || 0) > 0 ? 'GPU' : 'CPU'}), local`;
  if (sp.available) return 'speech: Windows recogniser (SAPI, local). faster-whisper is not installed (better for Portuguese names) - see `annotate asr-check`.';
  return 'no speech recogniser available: type labels, use M taps, or import a transcript JSON.';
}
async function pollAsr() {
  for (let i = 0; i < 60; i++) {
    const a = await api('GET', '/api/asr').catch(() => null);
    if (a && (a['faster-whisper'] || a.sapi)) { $('asrStatus').textContent = asrText(a); return; }
    await sleep(1000);
  }
}
async function boot() {
  wire();
  const info = await api('GET', '/api/session');
  S.info = info;
  document.title = `Move Narration - ${info.name}`;
  $('sessName').textContent = info.name;
  $('sessMode').textContent = info.mode === 'take' ? 'take - reference time' : 'video - video time';
  setMoves(info.moves);
  for (const c of info.cameras) S.cams[c.id] = c;
  $('camSel').innerHTML = info.cameras.map((c) => `<option value="${esc(c.id)}">${esc(c.id)} ${esc(c.name)}${c.playable === false ? ' (' + esc(c.codec) + ')' : ''}</option>`).join('');
  S.grid = info.beats || [];
  S.proposals = info.proposals;
  S.cb = info.counterbalance || [];
  S.segs = (info.labels && info.labels.segments) || [];
  sortSegs();
  if (S.grid.length > 1) {  // a timeline before (or without) the video's metadata
    S.range = [S.grid[0][0], S.grid[S.grid.length - 1][0]];
    S.view = S.segs.length ? { t0: S.segs[0].t0 - 5, t1: S.segs[S.segs.length - 1].t1 + 5 } : { t0: S.range[0], t1: S.range[1] };
  }
  const d = info.defaults || {};
  const saved = recall('settings', {});
  S.settings.lag = saved.lag != null ? saved.lag : (d.lag != null ? d.lag : 0.4);
  S.settings.snap = saved.snap || d.snap || 'half';
  S.settings.engine = saved.engine || d.engine || 'auto';
  S.settings.typesafe = saved.typesafe != null ? saved.typesafe : !!d.typesafe;
  $('lag').value = S.settings.lag; $('snap').value = S.settings.snap; $('engine').value = S.settings.engine;
  $('typesafe').checked = S.settings.typesafe;
  setRate(recall('speed', 0.5));
  if (!S.grid.length) { $('snap').value = 'none'; $('snap').title = 'no beat grid for this video'; }
  $('btnExport').disabled = !info.export_enabled;
  const notes = [];
  if (info.labels && info.labels.placeholder) notes.push(`labels.json holds ${info.labels.placeholder.count} PLACEHOLDER segments (not an analysis): they are not loaded and will be replaced when you save.`);
  for (const w of (info.labels && info.labels.warnings) || []) notes.push(esc(w));
  if (info.draft && info.draft.newer_than_labels && (info.draft.segments || []).length) notes.push(`An autosaved draft (${info.draft.segments.length} segments, ${esc(info.draft.saved)}) is newer than labels.json. <button id="btnDraft">Restore draft</button>`);
  banner(notes.join('<br>'));
  if ($('btnDraft')) $('btnDraft').onclick = () => { snapshot(); S.segs = info.draft.segments; changed(); banner(''); };
  $('asrStatus').textContent = asrText(info.asr);
  if (!info.asr) pollAsr();
  await refreshNarrationList();
  loadCamera(recall('camera.' + info.name, info.default_camera) in S.cams ? recall('camera.' + info.name, info.default_camera) : info.default_camera);
  $('camSel').addEventListener('change', (e) => store('camera.' + info.name, e.target.value));
  renderAll();
  requestAnimationFrame(frame);
  await deepLink();
}
async function deepLink() {
  // ?narration=<id>[&apply=1][&t=<label s>][&view=<t0>,<t1>][&sel=<segment number>] - reopen a narration from a link
  const q = new URLSearchParams(location.search);
  if (![...q.keys()].length) return;
  for (let i = 0; i < 100 && !S.view; i++) await sleep(50);
  if (q.get('narration')) await openNarration(q.get('narration'));
  if (q.get('apply') === '1' && S.narr) await applyNarration();
  if (q.get('view')) { const [a, b] = q.get('view').split(',').map(Number); if (b > a) S.view = { t0: a, t1: b }; }
  if (q.get('sel') && S.segs[+q.get('sel') - 1]) S.sel = S.segs[+q.get('sel') - 1].id;
  if (q.get('t')) seekT(+q.get('t'));
  renderAll();
}
boot().catch((e) => { banner('Could not start: ' + esc(e.message)); });
