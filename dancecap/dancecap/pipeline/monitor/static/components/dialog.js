// Modal dialogs, confirmations (incl. typed confirmation for pod deletion) and toasts.
import { h, replace } from "../util.js";

// Open dialogs, oldest first. Escape closes only the topmost one (one shared capture listener: a nested
// "Record consent..." dialog must not take the Plan & run dialog under it down with it).
const stack = [];

/** Close every open dialog (route changes: a dialog never outlives the view that opened it). */
export function closeAllDialogs() {
  for (const close of stack.slice().reverse()) close();
}

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || !stack.length) return;
  e.stopImmediatePropagation();
  e.preventDefault();
  stack[stack.length - 1]();
}, true);

/**
 * openDialog({title, body, actions, wide, onClose}) -> {el, body, close, setActions, busy}
 * actions: [{label, kind: "primary"|"danger"|"", onClick: async () => (false keeps the dialog open), disabled,
 *            autofocus}] -- the autofocus action (else the first input, else the primary button) gets the focus.
 */
export function openDialog({ title, body, actions = [], wide = false, onClose, className } = {}) {
  const prevFocus = document.activeElement;
  const bodyEl = h("div", { class: "dlg-body" }, body);
  const foot = h("div", { class: "dlg-foot" });
  const titleId = "dlg-t-" + Math.random().toString(36).slice(2, 8);
  const box = h("div", { class: ["dlg", wide ? "wide" : "", className], role: "dialog", "aria-modal": "true",
    "aria-labelledby": titleId, tabindex: "-1" },
    h("div", { class: "dlg-head" }, h("h2", { id: titleId }, title),
      h("button", { class: "icon-btn", title: "Close (Esc)", "aria-label": "Close", onclick: () => close() }, "✕")),
    bodyEl, foot);
  const back = h("div", { class: "dlg-back", onmousedown: (e) => { if (e.target === back) close(); } }, box);
  let closed = false;
  function close(result) {
    if (closed) return;
    closed = true;
    const i = stack.indexOf(close);
    if (i >= 0) stack.splice(i, 1);
    back.remove();
    if (prevFocus && prevFocus.focus && prevFocus.isConnected) prevFocus.focus({ preventScroll: true });
    if (onClose) onClose(result);
  }
  function setActions(acts) {
    replace(foot, acts.map((a) => {
      if (a.node) return a.node;
      const b = h("button", { class: ["btn", a.kind || ""], disabled: !!a.disabled, title: a.title || null,
        "data-autofocus": a.autofocus ? "1" : null }, a.label);
      b.addEventListener("click", async () => {
        if (b.disabled) return;
        if (!a.onClick) { close(); return; }
        b.disabled = true;
        b.classList.add("busy");
        try {
          const r = await a.onClick();
          if (r !== false) close(r);
        } catch (err) {
          toast(errorText(err), "crit");
        } finally {
          b.disabled = !!a.disabled;
          b.classList.remove("busy");
        }
      });
      return b;
    }));
  }
  setActions(actions);
  document.body.appendChild(back);
  stack.push(close);
  setTimeout(() => {
    if (closed) return;
    const f = box.querySelector("[autofocus], [data-autofocus]") || box.querySelector("input, select, textarea") || box.querySelector(".btn.primary");
    (f || box).focus?.({ preventScroll: true });
  }, 0);
  return { el: box, body: bodyEl, close, setActions, setBody: (...kids) => replace(bodyEl, ...kids) };
}

export const dialogOpen = () => stack.length > 0;

/**
 * Every confirmation here changes something (a take TOML, stage freshness, a run, the server), so Cancel has the
 * focus: a stray Enter / Space never confirms. Pass {defaultOk: true} only for harmless confirmations.
 */
export function confirmDialog(title, message, { okLabel = "OK", danger = false, cancelLabel = "Cancel", defaultOk = false } = {}) {
  return new Promise((resolve) => {
    let answered = false;
    openDialog({
      title, body: h("div", { class: "confirm-msg" }, message),
      actions: [
        { label: cancelLabel, autofocus: !defaultOk, onClick: () => { answered = true; resolve(false); } },
        { label: okLabel, kind: danger ? "danger" : "primary", autofocus: defaultOk, onClick: () => { answered = true; resolve(true); } },
      ],
      onClose: () => { if (!answered) resolve(false); },
    });
  });
}

/** Resolves with the typed text when it equals `expected` and the user confirms; null otherwise. */
export function typedConfirm(title, message, expected, { okLabel = "Delete", hint } = {}) {
  return new Promise((resolve) => {
    let answered = false;
    const input = h("input", { type: "text", class: "mono", autocomplete: "off", spellcheck: "false",
      placeholder: hint || expected, "aria-label": "confirmation" });
    const dlg = openDialog({
      title,
      body: h("div", { class: "confirm-msg" }, message,
        h("label", { class: "typed" }, h("span", null, "Type ", h("code", null, expected), " to confirm"), input)),
      actions: [
        { label: "Cancel", onClick: () => { answered = true; resolve(null); } },
        { label: okLabel, kind: "danger", disabled: true, onClick: () => { answered = true; resolve(input.value.trim()); } },
      ],
      onClose: () => { if (!answered) resolve(null); },
    });
    input.addEventListener("input", () => {
      const btn = dlg.el.querySelector(".dlg-foot .btn.danger");
      if (btn) btn.disabled = input.value.trim() !== expected;
    });
    input.focus();
  });
}

let toastHost = null;
export function toast(msg, kind = "info", { timeout = 6000, action } = {}) {
  if (!toastHost) {
    toastHost = h("div", { class: "toasts", role: "status", "aria-live": "polite" });
    document.body.appendChild(toastHost);
  }
  const icon = { good: "✓", crit: "✕", warn: "!", info: "i" }[kind] || "i";
  const t = h("div", { class: ["toast", "toast-" + kind] },
    h("span", { class: "toast-icon", "aria-hidden": "true" }, icon),
    h("div", { class: "toast-msg" }, msg),
    action ? h("button", { class: "btn small", onclick: () => { action.onClick(); t.remove(); } }, action.label) : null,
    h("button", { class: "icon-btn", "aria-label": "Dismiss", onclick: () => t.remove() }, "✕"));
  toastHost.appendChild(t);
  if (timeout) setTimeout(() => t.remove(), kind === "crit" ? Math.max(timeout, 12000) : timeout);
  return t;
}

/** A readable message for an ApiError (incl. 422 problems and gate details). */
export function errorText(err) {
  if (!err) return "Unknown error";
  if (err.name === "AbortError") return "Cancelled";
  if (err instanceof TypeError) return "The monitor backend is not answering (it may be restarting).";
  const parts = [err.message || String(err)];
  const d = err.details || {};
  if (Array.isArray(d.problems) && d.problems.length) {
    parts.push(d.problems.map((p) => (p.field ? `${p.field}: ` : "") + (p.message || "")).join("; "));
  }
  if (d.owner) parts.push(`held by: ${d.owner}`);
  return parts.join(" — ");
}
