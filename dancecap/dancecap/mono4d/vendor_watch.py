"""TypeSafe (Jev) readiness watch of the 4D providers: what do their own pages state today?

The route needs a provider that (1) takes ONE ordinary phone video of moving people, (2) returns a time-varying
reconstruction, (3) can be called by us (self-serve API, or access granted), and (4) gives renders at our cameras or a
file we can render ourselves. Vendors change fast (Atlas was announced 2026-09-01 as early access; Gracia moved from
"volumetric video" to "world reconstruction models"), so this re-reads their pages and asks, per page and capability:

    <cap>_stated      Noul: do these passages state that <vendor> <claim>?
    <cap>_ruled_out   Noul: do they state the opposite (<against>)? (only where a limitation is plausible)
    <cap>_evidence    Choice over the passage ids (+ "none"): which passage bears most directly on it?

Code owns the rest: a capability is "yes" when stated >= 0.7 on some page and not ruled out (>= 0.6 on some page),
"no" when ruled out, else "unclear"; a vendor is "ready" (for the 4D-provider route) when monocular_input,
dynamic_output and self_serve_api are all "yes" and camera_control or export is. Since 2026-10-19 the watch also covers
the tools proposed for the owner's end goal ("highly realistic 3D moving dancers with correct poses and natural
appearance, clothes, hair ... from monocular video"): the Atlas beta docs, NVIDIA Kimodo, Dense4D, Viggle JST-2 and the
Unity Learned Motion Matching asset, with five more capabilities (released, motion_from_video, clothing_hair,
motion_export, engine_playback). `_dancer_fit` counts the "yes" answers among DANCER_NEEDS: a ranking aid, not a
verdict (the measured tests in docs/MONO4D_NOVEL_VIEWS.md decide). Snapshots go to work/mono4d/vendor_watch/<date>.json with the evidence
passage of every answer, and each run reports what changed since the previous snapshot.

    python -m dancecap.mono4d vendors [--vendor gracia] [--no-save]
"""
from __future__ import annotations

import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

from typesafe_sdk import Choice, Noul

from ..take import PACKAGE_ROOT

OUT = PACKAGE_ROOT / "work" / "mono4d" / "vendor_watch"
MODEL = "jev-latest"
STATED, RULED_OUT = 0.7, 0.6
MAX_PAGE_CHARS = 30_000   # ~8k tokens: keep the state small and on topic (Jev's accuracy drops with big states)
PASSAGE_CHARS = 700

VENDORS = {
    "gracia": {"name": "Gracia", "urls": ["https://gracia.ai/4dgs", "https://gracia.ai/data",
                                          "https://gracia.ai/vfx-filmmaking"]},
    "worldlabs_atlas": {"name": "World Labs Atlas", "urls": ["https://www.worldlabs.ai/blog/atlas"]},
    "worldlabs_world_api": {"name": "the World Labs World API (Marble)", "urls": [
        "https://docs.worldlabs.ai/marble/create/prompt-guides/video-prompt.md",
        "https://docs.worldlabs.ai/api/index.md", "https://docs.worldlabs.ai/marble/create/prompt-guides/index.md"]},
    "worldlabs_atlas_beta": {"name": "the World Labs Atlas beta API (Marble 2)", "urls": [
        "http://atlas-beta.worldlabs.ai/docs/best-practices.md", "http://atlas-beta.worldlabs.ai/docs/how-to-use-atlas.md",
        "http://atlas-beta.worldlabs.ai/docs/atlas-generate.md"]},
    "nvidia_kimodo": {"name": "NVIDIA Kimodo", "urls": [
        "https://research.nvidia.com/labs/sil/projects/kimodo/",
        "https://raw.githubusercontent.com/nv-tlabs/kimodo/main/README.md"]},
    "dense4d": {"name": "Dense4D", "urls": ["https://alpguler.com/dense4d/", "https://arxiv.org/abs/2610.22325"]},
    "viggle_jst2": {"name": "Viggle JST-2", "urls": ["https://viggle.ai/jst2", "https://docs.viggle.ai/v1/introduction"]},
    "unity_lmm": {"name": "the Learned Motion Matching Unity asset (AiKodex)", "urls": [
        "https://assetstore.unity.com/packages/tools/animation/learned-motion-matching-312510",
        "https://www.aikodex.com/"]},
}

CAPABILITIES = {
    "monocular_input": {
        "claim": "accepts a single ordinary video recorded by one handheld camera, such as one phone, as enough input "
                 "to reconstruct people who are moving",
        "against": "requires several hardware-synchronized cameras (a multi-camera rig or genlock), or requires the "
                   "filmed scene to be static, without moving people"},
    "dynamic_output": {
        "claim": "produces a time-varying 3D reconstruction of moving people or objects, such as a 4D Gaussian splat "
                 "sequence or volumetric video",
        "against": "produces only static 3D scenes, or tells users to avoid moving people or objects"},
    "self_serve_api": {
        "claim": "offers a programmatic API with published endpoints that any developer can sign up for and call today",
        "against": "is available only through early access, a waitlist, select partners, a request form or by "
                   "contacting the company"},
    "camera_control": {
        "claim": "renders or generates views of the result from camera positions and orientations that the user "
                 "specifies exactly",
        "against": None},
    "export": {
        "claim": "lets the user download the reconstructed 3D or 4D result as files, such as PLY, SPZ or a dedicated "
                 "4D format",
        "against": None},
    "humans_in_motion": {
        "claim": "is shown or stated to work on real people performing in motion, such as dancers, athletes, actors "
                 "or musicians",
        "against": "tells users to avoid filming moving people, or says people are not yet a strength of its models"},
    "released": {
        "claim": "can be used today: its code and model weights are publicly downloadable, or a public product or API "
                 "is open to anyone",
        "against": "is not available yet: code or weights are coming soon, or access is only on request"},
    "motion_from_video": {
        "claim": "estimates or captures the 3D body motion of real people from ordinary video",
        "against": None},
    "clothing_hair": {
        "claim": "reconstructs, tracks or models people's clothing and hair, not only a skeleton or a body model",
        "against": None},
    "motion_export": {
        "claim": "outputs human motion as skeleton animation or body-model parameters (for example BVH, FBX, SMPL, "
                 "SMPL-X or SOMA) that a game engine can play",
        "against": None},
    "engine_playback": {
        "claim": "runs inside or plays back in a real-time engine such as Unity or Unreal, or on a VR headset",
        "against": None},
}
READY_NEEDS = ("monocular_input", "dynamic_output", "self_serve_api")
DANCER_NEEDS = ("monocular_input", "humans_in_motion", "motion_from_video", "clothing_hair", "released",
                "motion_export")


class _Text(HTMLParser):
    BLOCK = {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "br", "tr", "summary",
             "details", "dd", "dt", "blockquote", "figcaption"}

    SKIP = ("script", "style", "noscript", "svg", "nav", "footer", "form", "template")

    def __init__(self):
        super().__init__()
        self.parts, self.skip, self.ld, self._in_ld = [], 0, [], False

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("type") == "application/ld+json":
            self._in_ld = True
            self.ld.append("")
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_ld = False
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_ld:
            self.ld[-1] += data
        elif not self.skip:
            self.parts.append(data)


LD_KEYS = ("headline", "name", "description", "text", "articleBody", "abstract")


def _ld_lines(node, out: list[str]) -> None:
    """Readable strings of schema.org JSON-LD (client-rendered sites put their FAQ and descriptions only there)."""
    if isinstance(node, dict):
        for k in LD_KEYS:
            v = node.get(k)
            if isinstance(v, str) and len(v) > 2 and not v.startswith("http"):
                out.append(v)
        for k, v in node.items():
            if k not in LD_KEYS:
                _ld_lines(v, out)
    elif isinstance(node, list):
        for v in node:
            _ld_lines(v, out)


def page_text(raw: str, is_html: bool) -> str:
    """Visible text of a page (or a Markdown doc), plus the strings of its JSON-LD blocks, one line per block."""
    if is_html:
        p = _Text()
        p.feed(raw)
        ld: list[str] = []
        for block in p.ld:
            try:
                _ld_lines(json.loads(block), ld)
            except ValueError:
                continue
        visible = "".join(p.parts)
        raw = visible + "\n" + "\n".join(s for s in dict.fromkeys(ld) if s not in visible)
    raw = html.unescape(raw)
    lines = [re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def passages(text: str, size: int = PASSAGE_CHARS, limit: int = MAX_PAGE_CHARS) -> list[dict]:
    """Lines merged into passages of up to `size` characters, at most `limit` characters in all, ids p0, p1, ..."""
    out, cur = [], ""
    for ln in text[:limit].splitlines():
        if cur and len(cur) + len(ln) + 1 > size:
            out.append(cur)
            cur = ""
        cur = f"{cur}\n{ln}" if cur else ln
    if cur:
        out.append(cur)
    return [{"id": f"p{i}", "text": t} for i, t in enumerate(out)]


def fetch(url: str, http=None) -> str:
    import httpx
    get = (http or httpx).get
    r = get(url, follow_redirects=True, timeout=60.0, headers={"User-Agent": "dancecap-vendor-watch/1"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "text/html")
    text = r.text if "charset=" in ctype else r.content.decode("utf-8", errors="replace")  # no charset: UTF-8
    return page_text(text, is_html="html" in ctype and not url.endswith(".md"))


def questions(vendor_name: str, ids: list[str]) -> dict:
    q = {}
    for cap, c in CAPABILITIES.items():
        q[f"{cap}_stated"] = Noul(
            instructions=f"Do the passages in `passages` state that {vendor_name} {c['claim']}?",
            criteria={"true": "A passage says so directly, as a current capability or requirement.",
                      "false": "No passage says so, or it is only hinted at, planned, or about a different product."})
        if c["against"]:
            q[f"{cap}_ruled_out"] = Noul(
                instructions=f"Do the passages in `passages` state that {vendor_name} {c['against']}?",
                criteria={"true": "A passage states this requirement or limitation directly.",
                          "false": "No passage states it."})
        q[f"{cap}_evidence"] = Choice(
            instructions=f"Which passage in `passages` bears most directly on whether {vendor_name} {c['claim']}?",
            criteria={**{i: None for i in ids}, "none": "No passage is about this."})
    return q


def judge_page(vendor: str, url: str, text: str, client=None, model: str = MODEL) -> dict:
    if client is None:
        from ..judgments import client as _client
        client = _client()
    ps = passages(text)
    name = VENDORS[vendor]["name"]
    r = client.system_one(state={"vendor": name, "page_url": url, "passages": ps},
                          questions=questions(name, [p["id"] for p in ps]), model=model)
    a = r.answers
    by_id = {p["id"]: p["text"] for p in ps}
    caps = {}
    for cap, c in CAPABILITIES.items():
        ev = a[f"{cap}_evidence"]
        caps[cap] = {"stated": round(float(a[f"{cap}_stated"].noul), 3),
                     "ruled_out": round(float(a[f"{cap}_ruled_out"].noul), 3) if c["against"] else None,
                     "evidence_id": ev.choice, "evidence_conf": round(float(ev.confidence), 3),
                     "evidence": by_id.get(ev.choice, "")[:400]}
    return {"url": url, "passages": len(ps), "chars": len(text), "model": r.model,
            "input_tokens": int(getattr(r.usage, "input_tokens", 0) or 0), "capabilities": caps}


def decide(pages: list[dict]) -> dict:
    """Per capability over a vendor's pages -> "yes" | "no" | "unclear", with the deciding page and evidence."""
    out = {}
    for cap, c in CAPABILITIES.items():
        rows = [(p["url"], p["capabilities"][cap]) for p in pages if "capabilities" in p]
        if not rows:
            out[cap] = {"status": "unclear", "why": "no page judged"}
            continue
        url_s, best_s = max(rows, key=lambda r: r[1]["stated"])
        ro = [r for r in rows if r[1]["ruled_out"] is not None]
        url_r, best_r = max(ro, key=lambda r: r[1]["ruled_out"]) if ro else (None, None)
        if best_r is not None and best_r["ruled_out"] >= RULED_OUT:
            out[cap] = {"status": "no", "p": best_r["ruled_out"], "url": url_r, "evidence": best_r["evidence"]}
        elif best_s["stated"] >= STATED:
            out[cap] = {"status": "yes", "p": best_s["stated"], "url": url_s, "evidence": best_s["evidence"]}
        else:
            out[cap] = {"status": "unclear", "p": best_s["stated"], "url": url_s, "evidence": best_s["evidence"]}
    out["_ready"] = (all(out[c]["status"] == "yes" for c in READY_NEEDS)
                     and any(out[c]["status"] == "yes" for c in ("camera_control", "export")))
    out["_dancer_fit"] = sum(out[c]["status"] == "yes" for c in DANCER_NEEDS)
    return out


def changes(prev: dict | None, cur: dict) -> list[str]:
    if not prev:
        return []
    msgs = []
    for v, d in cur["vendors"].items():
        pd = (prev.get("vendors") or {}).get(v, {}).get("decision", {})
        for cap in CAPABILITIES:
            a, b = (pd.get(cap) or {}).get("status"), d["decision"][cap]["status"]
            if a and a != b:
                msgs.append(f"{v}: {cap} {a} -> {b}")
        if pd and pd.get("_ready") != d["decision"]["_ready"]:
            msgs.append(f"{v}: ready {pd.get('_ready')} -> {d['decision']['_ready']}")
    return msgs


def latest_snapshot(out: Path = OUT) -> dict | None:
    snaps = sorted(out.glob("*.json"))
    return json.loads(snaps[-1].read_text()) if snaps else None


def run(vendors: list[str] | None = None, save: bool = True, client=None, http=None, out: Path = OUT) -> dict:
    prev = latest_snapshot(out)
    doc = {"format": "dancecap.mono4d.vendor_watch/1", "date": time.strftime("%Y-%m-%d"), "model": MODEL,
           "thresholds": {"stated": STATED, "ruled_out": RULED_OUT}, "vendors": {}}
    for v in vendors or list(VENDORS):
        pages = []
        for url in VENDORS[v]["urls"]:
            try:
                pages.append(judge_page(v, url, fetch(url, http), client=client))
            except Exception as e:  # a page that cannot be read is recorded, not guessed
                pages.append({"url": url, "error": f"{type(e).__name__}: {e}"})
        doc["vendors"][v] = {"pages": pages, "decision": decide(pages)}
    doc["changes"] = changes(prev, doc)
    if save:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{time.strftime('%Y-%m-%d_%H%M%S')}.json").write_text(json.dumps(doc, indent=1))
    return doc


ABBR = {"monocular_input": "mono", "dynamic_output": "4D", "self_serve_api": "api", "camera_control": "cam",
        "export": "export", "humans_in_motion": "people", "released": "released", "motion_from_video": "mocap",
        "clothing_hair": "cloth/hair", "motion_export": "anim-out", "engine_playback": "engine"}


def table(doc: dict) -> str:
    """One row per vendor: each capability as yes (Y) / no (n) / unclear (?) with its P, then ready and dancer fit."""
    caps = list(CAPABILITIES)
    mark = {"yes": "Y", "no": "n", "unclear": "?"}
    rows = ["vendor".ljust(22) + "".join(ABBR.get(c, c)[:10].ljust(11) for c in caps) + "ready  dancer"]
    for v, d in doc["vendors"].items():
        dec = d["decision"]
        cells = [f"{mark[dec[c]['status']]} {dec[c].get('p', 0) or 0:.2f}".ljust(11) for c in caps]
        rows.append(v[:21].ljust(22) + "".join(cells) + f"{str(dec['_ready']):7}{dec['_dancer_fit']}/{len(DANCER_NEEDS)}")
    return "\n".join(rows)
