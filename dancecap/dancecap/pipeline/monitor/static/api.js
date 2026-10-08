// Fetch wrapper for the dancecap.monitor/1 API (PIPELINE_MONITOR.md 7): the per-instance token header on every
// write, ETag caching on reads, the error envelope turned into ApiError.

const meta = document.querySelector('meta[name="dancecap-token"]');
export const TOKEN = meta ? meta.getAttribute("content") : null;

export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message || code || `HTTP ${status}`);
    this.status = status;
    this.code = code || "error";
    this.details = details || {};
  }
}

const etagCache = new Map(); // path -> {etag, data}

async function parse(resp) {
  const ct = resp.headers.get("Content-Type") || "";
  if (ct.includes("application/json")) {
    try { return await resp.json(); } catch { return null; }
  }
  return await resp.text();
}

/**
 * A 403 that means "this window's credentials belong to an earlier server" (the token in the page or the session
 * cookie), as opposed to a business refusal (budget, consent, a typed confirmation). The server marks these with
 * details.auth; older servers only say so in the message.
 */
export function isAuthError(err) {
  if (!err || err.status !== 403 || err.code !== "forbidden") return false;
  if (err.details && err.details.auth) return true;
  return /X-Monitor-Token|session cookie|no monitor token/i.test(err.message || "");
}

function toError(resp, body) {
  const e = body && typeof body === "object" && body.error ? body.error : null;
  const err = new ApiError(resp.status, e ? e.code : "http_" + resp.status,
    e ? e.message : (typeof body === "string" && body ? body.slice(0, 200) : `HTTP ${resp.status}`),
    e ? e.details : {});
  if (isAuthError(err)) {
    try { window.dispatchEvent(new CustomEvent("dcm:auth-expired", { detail: { message: err.message } })); } catch { /* no window */ }
  }
  return err;
}

/**
 * GET with ETag reuse. Returns {data, changed}. `changed` is false when the server answered 304 (or the body is
 * byte-identical to the cached one), so views can skip a re-render.
 */
export async function getx(path, { signal } = {}) {
  const c = etagCache.get(path);
  const headers = { Accept: "application/json" };
  if (c && c.etag) headers["If-None-Match"] = c.etag;
  const resp = await fetch(path, { headers, credentials: "same-origin", cache: "no-store", signal });
  if (resp.status === 304 && c) return { data: c.data, changed: false };
  const body = await parse(resp);
  if (!resp.ok) throw toError(resp, body);
  const etag = resp.headers.get("ETag");
  if (etag) etagCache.set(path, { etag, data: body });
  return { data: body, changed: true };
}

export async function get(path, opts) {
  return (await getx(path, opts)).data;
}

export async function send(method, path, body, { ifMatch, signal } = {}) {
  if (!TOKEN) throw new ApiError(403, "forbidden", "No monitor token on this page: open the monitor from its own window.");
  const headers = { "X-Monitor-Token": TOKEN, Accept: "application/json" };
  let payload;
  if (body !== undefined) {
    headers["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }
  if (ifMatch) headers["If-Match"] = `"${String(ifMatch).replace(/"/g, "")}"`;
  const resp = await fetch(path, { method, headers, body: payload, credentials: "same-origin", cache: "no-store", signal });
  const data = await parse(resp);
  if (!resp.ok) throw toError(resp, data);
  return data;
}

export const post = (p, b, o) => send("POST", p, b === undefined ? {} : b, o);
export const put = (p, b, o) => send("PUT", p, b, o);
export const patch = (p, b, o) => send("PATCH", p, b, o);

export function forget(prefix) {
  for (const k of Array.from(etagCache.keys())) if (k.startsWith(prefix)) etagCache.delete(k);
}

/** Path segment encoder for ids that the server validates anyway (take, cam, stage, sid, run). */
export const seg = (x) => encodeURIComponent(String(x));

/** Some list endpoints answered a bare list in early backend builds and {items..., toml_etag} later. */
export function listOf(data, key) {
  if (Array.isArray(data)) return data;
  if (data && Array.isArray(data[key])) return data[key];
  return [];
}
