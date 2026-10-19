r"""Download the gated body / hair models the tool round needs, with YOUR registration logins (Python stdlib only).

    .venv-nopo4d\Scripts\python.exe tools\fetch_gated_models.py                  # all: smpl, smplify, difflocks
    .venv-nopo4d\Scripts\python.exe tools\fetch_gated_models.py --only smpl,smplify

Each item is one file behind the MPI-IS download server (https://download.is.tue.mpg.de/download.php), the same
endpoint the projects' own scripts call, but:
  - the password prompt is hidden (getpass) and the login goes only into the HTTPS POST body: never a command
    line, a file, a log or the screen; nothing is stored;
  - TLS certificates are verified (the official DiffLocks script turns that off);
  - a download resumes after an interruption (.part + Range), and the result must be a real zip (checked by its
    bytes: the server labels everything text/html, files included). A wrong login is HTTP 401 ("Username/Password
    wrong"); any other web page is shown by its message line and deleted;
  - zips are unpacked into models/<item>/ (git-ignored: the licences forbid redistribution) and recorded in
    models/INDEX.json.
Register (and accept each licence) first on the item's site; one account per site.
"""
from __future__ import annotations

import argparse
import getpass
import html
import json
import re
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
SERVER = "https://download.is.tue.mpg.de/download.php"
ITEMS = {
    "smpl": {"site": "https://smpl.is.tue.mpg.de/", "domain": "smpl", "file": "SMPL_python_v.1.1.0.zip",
             "why": "SMPL 1.1 (female / male / neutral .pkl): CloseApp and other SMPL-based tools"},
    "smplify": {"site": "https://smplify.is.tue.mpg.de/", "domain": "smplify", "file": "mpips_smplify_public_v2.zip",
                "why": "SMPLify release (its neutral SMPL model basicModel_neutral_lbs_10_207_0_v1.0.0.pkl): CloseApp"},
    "difflocks": {"site": "https://difflocks.is.tue.mpg.de/", "domain": "difflocks", "file": "difflocks_checkpoints.zip",
                  "why": "DiffLocks checkpoints (strand hair from one photo)"},
}
ERROR_LINE = re.compile(r"(Error:[^<\r\n]{0,200})")
TITLE = re.compile(r"<title>([^<]{0,120})</title>", re.IGNORECASE)
NOISE = re.compile(r"<(script|style|head)\b.*?</(script|style|head)>", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]+>")


def ask_login(item: dict, previous: tuple[str, str] | None) -> tuple[str, str] | None:
    print(f"\n== {item['why']}\n   account at {item['site']} (register and accept the licence there first)")
    if previous:
        reuse = input("   Use the same username and password as the previous site? [y/N]: ").strip().lower()
        if reuse in ("y", "yes"):
            return previous
    user = input("   Username / e-mail (Enter to skip this item): ").strip()
    if not user:
        return None
    pw = getpass.getpass("   Password (hidden; paste with RIGHT-CLICK, Ctrl+V may not paste here): ")
    if any(ord(c) < 32 for c in pw):  # Ctrl+V at a hidden prompt arrives as a control character, not the clipboard
        print("   the password contains a control character (Ctrl+V?): re-enter it, pasting with right-click")
        pw = getpass.getpass("   Password (hidden): ")
    return (user, pw) if pw else None


def read_login_file(path: Path) -> dict[str, tuple[str, str]]:
    """YOUR notes file -> {item: (username, password)}. Format: a 'username: ...' line, then for each site its URL on
    one line and that site's password on the next non-empty line. Read only when you run this with --login-file."""
    user, site, logins = None, None, {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith(("username:", "user:", "email:", "e-mail:")):
            user = line.split(":", 1)[1].strip()
        elif low.startswith(("http://", "https://")):
            host = urllib.parse.urlparse(line).netloc.lower()
            site = next((n for n, it in ITEMS.items() if urllib.parse.urlparse(it["site"]).netloc == host), None)
        elif site and user:
            logins[site] = (user, line)
            site = None
    return logins


def log_line(msg: str) -> None:
    """Outcome log for diagnosis (models/.fetch_log.txt): what happened per item, never a login."""
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(MODELS / ".fetch_log.txt", "a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def page_message(data: bytes) -> str:
    """The server's own words from an HTML answer (its 'Error: ...' line or the page title), never the login."""
    text = data.decode("utf-8", errors="replace")
    m = ERROR_LINE.search(text)
    if m:
        return html.unescape(m.group(1)).strip()
    t = TITLE.search(text)
    words = " ".join(TAG.sub(" ", NOISE.sub(" ", text)).split())[:200]
    return (f"page '{html.unescape(t.group(1)).strip()}': " if t else "") + words


class _LoggedRedirects(urllib.request.HTTPRedirectHandler):
    """Follows redirects like wget (POST -> GET) and notes each hop as 'code host/path' (no query: it can hold
    tokens)."""

    def __init__(self, hops: list[str]):
        super().__init__()
        self.hops = hops

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        u = urllib.parse.urlparse(newurl)
        self.hops.append(f"{code} {u.netloc}{u.path}")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def opener(hops: list[str]) -> urllib.request.OpenerDirector:
    """TLS-verified opener WITH a cookie jar: after a good login the server redirects and expects the session cookie
    it just set (wget, which the projects' own scripts use, keeps cookies by default; plain urllib does not)."""
    import http.cookiejar
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl.create_default_context()),
                                       urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()),
                                       _LoggedRedirects(hops))


def download(item: dict, login: tuple[str, str], dest: Path) -> Path:
    """POST the login, stream the file to dest/<file>.part (resuming), check it is a zip -> the zip path."""
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / item["file"]
    part = out.with_name(out.name + ".part")
    if out.exists() and zipfile.is_zipfile(out):
        print(f"   already downloaded: {out}")
        return out
    url = f"{SERVER}?{urllib.parse.urlencode({'domain': item['domain'], 'sfile': item['file'], 'resume': 1})}"
    body = urllib.parse.urlencode({"username": login[0], "password": login[1]}).encode()
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/x-www-form-urlencoded",
                                          "User-Agent": "dancecap-fetch-gated-models/1",
                                          **({"Range": f"bytes={have}-"} if have else {})})
    hops: list[str] = []
    try:
        r = opener(hops).open(req, timeout=120)
    except urllib.error.HTTPError as e:
        msg = page_message(e.read(20000))
        if e.code == 401:
            raise PermissionError(f"the server refused the login ({msg}); check the username / password of "
                                  f"{item['site']}") from None
        if e.code == 416 and part.exists() and zipfile.is_zipfile(part):  # the .part already holds the whole file
            part.replace(out)
            return out
        raise RuntimeError(f"HTTP {e.code}: {msg}") from None
    if hops:
        log_line(f"{item['domain']}: redirects {' -> '.join(hops)}")
    with r:
        total = r.headers.get("Content-Length")
        resumed = r.status == 206
        got = have if resumed else 0
        size = (int(total) + got) if total else None
        t0, last = time.time(), 0.0
        with open(part, "ab" if resumed else "wb") as fh:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if time.time() - last > 1.0:
                    rate = (got - (have if resumed else 0)) / max(time.time() - t0, 1e-6) / 1e6
                    pct = f" {100 * got / size:5.1f} %" if size else ""
                    print(f"\r   {got / 1e6:8.1f} MB{pct}  {rate:5.1f} MB/s", end="", flush=True)
                    last = time.time()
    print(f"\r   {got / 1e6:8.1f} MB downloaded{' ' * 20}")
    if not zipfile.is_zipfile(part):  # decided by the bytes: the server labels the files text/html too
        data = part.read_bytes()[:20000]
        part.unlink()
        if b"<html" in data.lower() or b"<!doctype" in data.lower():
            via = f" after redirects {' -> '.join(hops)}" if hops else " (no redirect)"
            raise PermissionError(f"the server sent a web page instead of the file{via}: {page_message(data)} "
                                  f"(the login itself was accepted - a wrong one is HTTP 401; check that the "
                                  f"account is activated and the licence accepted on {item['site']})")
        raise ValueError("the download is not a zip file (deleted); try again later")
    part.replace(out)
    return out


def unpack(zpath: Path, dest: Path) -> list[str]:
    with zipfile.ZipFile(zpath) as z:
        bad = [n for n in z.namelist() if n.startswith(("/", "\\")) or ".." in Path(n).parts]
        if bad:
            raise ValueError(f"refusing unsafe paths in {zpath.name}: {bad[:3]}")
        z.extractall(dest)
        return [n for n in z.namelist() if not n.endswith("/")]


def record(name: str, item: dict, files: list[str], dest: Path) -> None:
    idx_p = MODELS / "INDEX.json"
    idx = json.loads(idx_p.read_text()) if idx_p.exists() else {}
    keep = [f for f in files if f.lower().endswith((".pkl", ".npz", ".pt", ".pth", ".ckpt", ".safetensors"))]
    idx[name] = {"dir": dest.relative_to(ROOT).as_posix(), "source": item["site"], "archive": item["file"],
                 "model_files": keep[:40], "downloaded": time.strftime("%Y-%m-%d"),
                 "licence": "gated; registration licence of the site - never commit or redistribute"}
    idx_p.write_text(json.dumps(idx, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated: " + ", ".join(ITEMS))
    ap.add_argument("--login-file", help="your own notes file with the username and each site's password (see "
                                         "read_login_file); no prompts")
    a = ap.parse_args(argv)
    names = [n.strip() for n in a.only.split(",")] if a.only else list(ITEMS)
    unknown = [n for n in names if n not in ITEMS]
    if unknown:
        ap.error(f"unknown items {unknown}; choose from {list(ITEMS)}")
    from_file = read_login_file(Path(a.login_file)) if a.login_file else None
    if from_file is not None:
        print(f"logins found in the file for: {sorted(from_file) or 'none'}")
    previous, done, failed = None, [], []
    for name in names:
        item = ITEMS[name]
        dest = MODELS / name
        if from_file is not None:
            print(f"\n== {item['why']}")
            login = from_file.get(name)
        else:
            login = ask_login(item, previous)
        if login is None:
            print("   skipped (no login)")
            log_line(f"{name}: skipped (no login)")
            continue
        try:
            z = download(item, login, dest)
            files = unpack(z, dest)
            record(name, item, files, dest)
            done.append(name)
            print(f"   ok: {len(files)} files in {dest}")
            log_line(f"{name}: ok, {len(files)} files")
            previous = login
        except Exception as e:  # never echo the login: the messages name the site and the file only
            failed.append(name)
            print(f"   FAILED ({type(e).__name__}): {e}")
            log_line(f"{name}: FAILED {type(e).__name__}: {e}")
    print(f"\nDone: {done or 'none'}; failed: {failed or 'none'}. Files stay in {MODELS} (git-ignored).")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
