#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""hm-guard: shared commit guard for the HeadMovement repositories (stdlib only).

One identical copy of this file lives in each repository (workspace: tools/githooks/,
Unity: Tools/githooks/).  The hooks `pre-commit` and `commit-msg` call it with a profile.

  --staged            check what is staged (the git index): the pre-commit mode
  --commit-msg FILE   check a commit message file: the commit-msg mode
  --pre-push REMOTE URL [--stdin-file F]
                      the pre-push mode: reads the ref lines git writes to the hook's stdin,
                      refuses deletions and non-fast-forward updates, audits every commit that
                      would be sent (messages and added lines, as at commit time), and, in the
                      workspace profile, refuses every push while the remote is treated as public
  --worktree [PATH..] audit working-tree files as if they were about to be added
                      (default: every tracked and untracked-not-ignored file)
  --self-test         run the built-in decoy tests in a throw-away repository
  --version           print the version and the sha256 of this file

Rules (ids are printed in brackets):
  key-file      key / token / .env / credential file names, the private terms file itself
  gated-model   gated body-model files (SMPL-X, SMPL, FLAME, MANO ...)
  model-data    weights, pickles, arrays, point clouds, splats
  media         audio and video files
  binary-asset  NEW image or 3D-model files (they may carry a likeness) unless allow-listed
  likeness      file names that look like portraits, textures or grooms of a person
  private-path  files under paths that are git-ignored on purpose: capture data
                (Assets/StreamingAssets), recordings, run outputs (work/); never allow-listable
  size          files above the profile limit unless allow-listed
  secret        secret patterns in added lines, exact values of the local key files
  term          (unity profile) a line or path containing an entry of the private terms file;
                (workspace profile, public remote) a net increase of such lines in a file that
                terms-allow.txt does not list: a warning, an error with hm.guard.terms=block
  local-path    an absolute path that contains a user directory: every added line in the unity
                profile; a net increase per file in the workspace profile while the remote is public
  push          (pre-push) a deletion, a non-fast-forward update, or any push at all from the
                workspace repository while its remote is treated as public

The text of a private term is never printed: hits are reported as "term #N" where N is
the line number of the entry in the private terms file, and paths are masked.
A line containing the marker "hm-guard: ignore" is exempt from the pattern based secret
and local-path findings (never from exact key values or terms).

Workspace profile and the remote's visibility: the workspace remote answered unauthenticated
requests on 2026-10-08, so it is PUBLIC.  Until the owner has made the GitHub repository
private and recorded that with `git config hm.visibility private`, the guard treats it as
public: no push at all (pre-push), absolute user paths cannot be added, and growth of the
private terms in files outside terms-allow.txt is reported.  Environment overrides:
HM_GUARD_VISIBILITY=public|private, HM_GUARD_TERMS_MODE=off|warn|block (git config hm.guard.terms).

Exit status: 0 = clean, 1 = violations, 2 = internal error.  Emergency bypass (repo owner
only): see GIT.md; it is implemented in the hook scripts, not here.
"""
from __future__ import annotations

import argparse
import fnmatch
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata

VERSION = "1.1"
MB = 1024 * 1024
ZERO_SHA = "0" * 40

PROFILES = {
    # workspace: pipeline code and small configs.  Dancer specific data is expected here, so
    # the private term list is NOT applied line by line (opt in with HM_GUARD_TERMS=1).  While the
    # remote is public ("ratchet"): the number of lines with private terms or absolute user paths
    # in a file may not grow (terms: a warning by default), and nothing may be pushed.
    "workspace": {"max_bytes": 5 * MB, "terms": False, "generic": False, "ratchet": True},
    # unity: public repository.  Private terms and absolute user paths are rejected.
    "unity": {"max_bytes": 5 * MB, "terms": True, "generic": True, "ratchet": False},
}

# Paths that are git-ignored on purpose; the ignore file alone is not enough (git add -f).  Matched on the
# lower-cased path (Windows file systems are case-insensitive).  (regex, exception regex or None, message)
PATH_RULES = {
    "unity": [
        (re.compile(r"^assets/streamingassets/"), re.compile(r"^assets/streamingassets/\.gitkeep(\.meta)?$"),
         "capture data under Assets/StreamingAssets/ is git-ignored on purpose (only .gitkeep may be committed)"),
        (re.compile(r"^recordings/"), None,
         "recordings and film renders (Recordings/) contain copyrighted audio and personal data and are never committed"),
    ],
    "workspace": [
        (re.compile(r"^dancecap/work/|^atlas_bridge/sessions/[^/]+/work/"), None,
         "run outputs under work/ are git-ignored on purpose (analysis data, footage derivatives)"),
        (re.compile(r"^(?:models|weights|smplx|dances|footage|recordings)/"), None,
         "body models, weights, footage and recordings stay outside git (copied to pods from the local machine)"),
    ],
}

# ---------------------------------------------------------------------------- name rules
KEY_NAME_RES = [
    re.compile(r"(^|/)[^/]*-key\.txt$"),
    re.compile(r"\.(key|pem|p12|pfx|jks|keystore|token|secret|secrets|ppk)$"),
    re.compile(r"(^|/)\.env($|\.)(?!example$|sample$|template$)"),
    re.compile(r"(^|/)id_(rsa|dsa|ecdsa|ed25519)$"),
    re.compile(r"(^|/)(credentials|secrets?)(\.(json|toml|ya?ml|ini|cfg|txt|env))?$"),
    re.compile(r"(^|/)\.(netrc|npmrc|pypirc|git-credentials)$"),
    re.compile(r"(^|/)(service[-_]account|client[-_]secret)[^/]*\.json$"),
    re.compile(r"(^|/)\.hm-forbidden-terms[^/]*$"),
]

GATED_TOKEN_RE = re.compile(r"(?:^|[/_\-. ])(?:smpl[-_ ]?x|smplx|smpl[-_ ]?h|smplh|smpl|flame|mano|supr)(?:[/_\-. 0-9]|$)")
GATED_EXTS = {".obj", ".ply", ".fbx", ".glb", ".gltf", ".bin", ".dat", ".pkl", ".pickle", ".npz", ".npy", ".pt",
              ".pth", ".ckpt", ".safetensors", ".h5", ".hdf5", ".onnx", ".blend", ".vrm", ".usd", ".usdz"}

MODEL_DATA_EXTS = {".npz", ".npy", ".pkl", ".pickle", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".h5",
                   ".hdf5", ".joblib", ".tflite", ".engine", ".trt", ".weights", ".caffemodel"}
CAPTURE_EXTS = {".ply", ".splat", ".spz", ".pcd", ".las", ".laz", ".e57"}

MEDIA_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".oga", ".opus", ".wma", ".aiff", ".aif",
              ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".mts"}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".jpe", ".jfif", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".exr", ".tga",
              ".hdr", ".psd", ".psb", ".heic", ".heif", ".dds", ".ktx", ".ktx2", ".avif", ".xcf", ".kra",
              ".cr2", ".nef", ".arw", ".dng"}
ASSET3D_EXTS = {".obj", ".fbx", ".glb", ".gltf", ".blend", ".vrm", ".usd", ".usdz", ".abc", ".dae", ".3ds", ".stl"}

CODE_EXTS = {".py", ".cs", ".js", ".mjs", ".ts", ".css", ".html", ".ps1", ".cmd", ".bat", ".sh", ".md", ".rst",
             ".shader", ".cginc", ".hlsl", ".compute", ".asmdef", ".asmref", ".slnx", ".csproj", ".gitignore",
             ".gitattributes"}

LIKENESS_NAME_RE = re.compile(
    r"(?:portrait|likeness|head[-_ ]?shot|selfie|face[-_ ]?(?:photo|scan|tex(?:ture)?|ref(?:erence)?|shape)"
    r"|skin[-_ ]?(?:tex(?:ture)?|map)|groom)", re.I)

ALLOWABLE_RULES = ("size", "binary-asset")

# ---------------------------------------------------------------------- content patterns
_PLACEHOLDER_RE = re.compile(r"(?i)(example|placeholder|your|changeme|change_me|dummy|sample|redacted|xxxx|\*\*\*|\.\.\.|<|>|\$\{|%s|\{\{)")


def _assignment_ok(m):
    v = m.group("v")
    return bool(re.search(r"[A-Za-z]", v)) and bool(re.search(r"\d", v)) and not _PLACEHOLDER_RE.search(v)


def _url_cred_ok(m):
    pw = m.group("pw")
    return not _PLACEHOLDER_RE.search(pw) and pw.lower() not in ("pass", "password", "passwd", "secret", "token", "user")


SECRET_PATTERNS = [
    ("OpenAI/Anthropic-style API key", re.compile(r"\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{20,}"), None),
    ("Runpod-style API key", re.compile(r"\brpa_[A-Za-z0-9]{20,}"), None),
    ("bearer token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/\-]{20,}"), None),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), None),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}|\bgithub_pat_[A-Za-z0-9_]{30,}"), None),
    ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), None),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"), None),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"), None),
    ("private key block", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY"), None),
    ("JSON web token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"), None),
    ("credentials in a URL", re.compile(r"\b[a-z][a-z0-9+.\-]*://[^/\s:@'\"<>]{1,64}:(?P<pw>[^/\s@'\"<>]{4,})@[A-Za-z0-9.\-]+"), _url_cred_ok),
    ("secret-looking assignment", re.compile(
        r"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd|token|secret)\b[\"']?\s*[:=]\s*[\"'](?P<v>[A-Za-z0-9/+_\-.=]{16,})[\"']"), _assignment_ok),
]

_EXCL = r"(?:<|%|\$|\{|\[|\.\.\.|public\b|default\b|all users\b|you\b|your|user(?:name)?\b|name\b|me\b|x+\b|someone\b|example\b|shared\b|runner\b|ubuntu\b|root\b|vscode\b|ec2-user\b)"
LOCAL_PATH_RES = [
    re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]+Users[\\/]+(?!" + _EXCL + r")[^\\/\s\"'`<>|*?:]{2,}", re.I),
    re.compile(r"(?<![A-Za-z0-9_.~])/(?:Users|home)/(?!" + _EXCL + r")[^/\s\"'`<>]{2,}/"),
]

IGNORE_MARK = "hm-guard: ignore"


# --------------------------------------------------------------------------- utilities
def fold(s):
    """Lower-case-insensitive helper: strip accents so that variants still match."""
    if s.isascii():
        return s
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


class Ctx:
    def __init__(self, profile, root):
        self.profile = profile
        self.cfg = PROFILES[profile]
        self.root = root
        env = os.environ.get("HM_GUARD_TERMS")
        self.use_terms = self.cfg["terms"] if env is None else env.strip().lower() not in ("", "0", "false", "no", "off")
        self.terms_path = os.environ.get("HM_GUARD_TERMS_FILE") or os.path.join(os.path.expanduser("~"), ".hm-forbidden-terms.txt")
        self.warnings = []
        self.term_any = None
        self.term_each = []          # (file line number, compiled regex)
        self.secret_values = self._load_secret_values()
        self.allow = []
        self.terms_allow = []
        # visibility of the remote: "public" unless the owner recorded `git config hm.visibility private`
        vis = (os.environ.get("HM_GUARD_VISIBILITY") or self.git_config("hm.visibility") or "public").strip().lower()
        self.visibility = "private" if vis == "private" else "public"
        # ratchet mode (workspace profile, public remote): no growth of private terms / absolute user paths per file
        self.ratchet = bool(self.cfg.get("ratchet")) and self.visibility == "public" and not self.use_terms
        mode = (os.environ.get("HM_GUARD_TERMS_MODE") or self.git_config("hm.guard.terms") or "warn").strip().lower()
        self.terms_mode = mode if mode in ("off", "warn", "block") else "warn"
        self._load_terms()

    def git_config(self, key):
        try:
            p = subprocess.run(["git", "config", "--get", key], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return ""
        return p.stdout.decode("utf-8", "replace").strip() if p.returncode == 0 else ""

    # -- private terms ---------------------------------------------------------------
    def _load_terms(self):
        if not (self.use_terms or (self.ratchet and self.terms_mode != "off")):
            return
        try:
            with open(self.terms_path, encoding="utf-8-sig") as f:
                raw = f.read().splitlines()
        except OSError:
            self.warnings.append("private terms file not found (%s): the term check is SKIPPED" % self.terms_path)
            return
        sep = r"[\s_\-.'\u2019]*"
        for i, line in enumerate(raw, 1):
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            parts = [p for p in re.split(r"[\s_\-.'\u2019]+", fold(t)) if p]
            if not parts:
                continue
            if sum(len(p) for p in parts) < 3:
                self.warnings.append("term #%d is shorter than 3 characters and is ignored" % i)
                continue
            self.term_each.append((i, re.compile(sep.join(re.escape(p) for p in parts), re.I)))
        if self.term_each:
            self.term_any = re.compile("|".join("(?:%s)" % r.pattern for _, r in self.term_each), re.I)

    def term_hits(self, text):
        if self.term_any is None:
            return []
        t = fold(text)
        if not self.term_any.search(t):
            return []
        return [n for n, r in self.term_each if r.search(t)]

    def mask(self, text):
        """Return text with every private term replaced by <T#n> (for safe printing)."""
        if self.term_any is None:
            return text
        t = fold(text)
        for n, r in self.term_each:
            t = r.sub("<T%d>" % n, t)
        return t

    # -- exact key values --------------------------------------------------------------
    @staticmethod
    def _load_secret_values():
        home = os.path.expanduser("~")
        pats = [os.path.join(home, "Desktop", "*-key.txt"), os.path.join(home, "*-key.txt"),
                os.path.join(home, "OneDrive", "Desktop", "*-key.txt")]
        extra = os.environ.get("HM_GUARD_SECRET_FILES")
        if extra:
            pats += [p for p in extra.split(os.pathsep) if p]
        vals = set()
        for pat in pats:
            for fn in glob.glob(pat):
                try:
                    with open(fn, encoding="utf-8", errors="ignore") as f:
                        data = f.read(8192)
                except OSError:
                    continue
                for ln in data.splitlines():
                    ln = ln.strip()
                    if len(ln) >= 12 and " " not in ln:
                        vals.add(ln)
                    m = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*[:=]\s*(\S{12,})$", ln)
                    if m:
                        vals.add(m.group(1).strip("\"'"))
        return sorted(vals)

    # -- allow lists -------------------------------------------------------------------
    @staticmethod
    def _globs(text):
        out = []
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line.replace("\\", "/"))
        return out

    def load_allow(self, rel_path, text):
        self.allow = self._globs(text)

    def load_terms_allow(self, text):
        self.terms_allow = self._globs(text)

    def is_allowed(self, path):
        return any(fnmatch.fnmatchcase(path, pat) for pat in self.allow)

    def terms_allowed(self, path):
        return any(fnmatch.fnmatchcase(path, pat) for pat in self.terms_allow)


class Violation:
    __slots__ = ("rule", "path", "line", "msg")

    def __init__(self, rule, path, line, msg):
        self.rule, self.path, self.line, self.msg = rule, path, line, msg


class Entry:
    def __init__(self, path, status, size=None, sha=None, mode=None, lines=None, binary=False):
        self.path, self.status, self.size, self.sha, self.mode = path, status, size, sha, mode
        self.lines = lines or []
        self.removed = []            # text of the removed lines (ratchet mode compares added against removed)
        self.binary = binary

    @property
    def is_new(self):
        return self.status[:1] in ("A", "C", "R")


# ------------------------------------------------------------------------------ git glue
def git(ctx, *args, input_bytes=None, ok_fail=False):
    cmd = ["git", "-c", "core.quotepath=off"] + list(args)
    p = subprocess.run(cmd, cwd=ctx.root, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 and not ok_fail:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), p.stderr.decode("utf-8", "replace").strip()))
    return p.stdout if p.returncode == 0 else b""


def repo_root():
    # git runs hooks from the top of the working tree: skip a process when that is plainly the case
    if os.path.exists(os.path.join(os.getcwd(), ".git")):
        return os.getcwd()
    p = subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("not inside a git repository")
    return p.stdout.decode("utf-8", "replace").strip()


def unquote_git_path(p):
    if not (p.startswith('"') and p.endswith('"')):
        return p
    body = p[1:-1]
    out = bytearray()
    i = 0
    esc = {"n": 10, "t": 9, "r": 13, '"': 34, "\\": 92, "a": 7, "b": 8, "f": 12, "v": 11}
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            n = body[i + 1]
            if n in esc:
                out.append(esc[n])
                i += 2
                continue
            if n in "01234567":
                j = i + 1
                k = j
                while k < len(body) and k < j + 3 and body[k] in "01234567":
                    k += 1
                out.append(int(body[j:k], 8) & 255)
                i = k
                continue
        out.extend(c.encode("utf-8"))
        i += 1
    return out.decode("utf-8", "replace")


def staged_entries(ctx):
    return diff_entries(ctx, ["diff", "--cached"])


def commit_entries(ctx, sha):
    """The entries a single-parent (or root) commit introduces."""
    return diff_entries(ctx, ["diff-tree", "--root", "--no-commit-id", "-r", sha])


def diff_entries(ctx, base_args):
    """One git call: the raw entry list followed by the zero-context patch (added lines with numbers).
    base_args select what is compared: ["diff", "--cached"] (the index against HEAD) or a diff-tree call for a commit."""
    out = git(ctx, *base_args, "--patch-with-raw", "--abbrev=40", "-U0", "-M", "--no-color", "--no-ext-diff",
              "--no-textconv", "--diff-filter=ACMRT", "--src-prefix=a/", "--dst-prefix=b/")
    out_lines = out.split(b"\n")
    entries = []
    k = 0
    while k < len(out_lines) and out_lines[k].startswith(b":"):
        head = out_lines[k].decode("utf-8", "replace")
        k += 1
        meta, *paths = head.split("\t")
        parts = meta[1:].split()
        entries.append(Entry(unquote_git_path(paths[-1]), parts[4], sha=parts[3], mode=parts[1]))
    by_path = {e.path: e for e in entries}
    cur = None
    ln = 0
    old_left = new_left = 0
    for raw_line in out_lines[k:]:
        line = raw_line.decode("utf-8", "replace")
        if old_left > 0 or new_left > 0:
            if line.startswith("+"):
                new_left -= 1
                if cur is not None:
                    cur.lines.append((ln, line[1:].rstrip("\r")))
                ln += 1
            elif line.startswith("-"):
                old_left -= 1
                if cur is not None:
                    cur.removed.append(line[1:].rstrip("\r"))
            # "\ No newline at end of file" lines do not change the counters
            continue
        if line.startswith("diff --git "):
            cur = None
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            m = re.search(r" and b/(.+) differ$", line)
            tgt = by_path.get(unquote_git_path(m.group(1))) if m else cur
            if tgt is not None:
                tgt.binary = True
        elif line.startswith("+++ "):
            p = line[4:].rstrip("\t").rstrip("\r")
            if p == "/dev/null":
                cur = None
            else:
                p = unquote_git_path(p)
                p = p[2:] if p.startswith("b/") else p
                cur = by_path.get(p)
        elif line.startswith("@@"):
            m = re.match(r"@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
            if m:
                old_left = int(m.group(1)) if m.group(1) is not None else 1
                ln = int(m.group(2))
                new_left = int(m.group(3)) if m.group(3) is not None else 1
    # sizes: new text files from their added lines; binary files with one cat-file call (only when there are any)
    binaries = []
    for e in entries:
        if e.mode == "160000":
            continue
        if e.binary:
            binaries.append(e)
        elif e.is_new:
            e.size = sum(len(s.encode("utf-8", "replace")) + 1 for _, s in e.lines)
    if binaries:
        res = git(ctx, "cat-file", "--batch-check=%(objectname) %(objectsize)",
                  input_bytes=("\n".join(e.sha for e in binaries) + "\n").encode())
        sizes = {}
        for sz in res.decode("utf-8", "replace").splitlines():
            a = sz.split()
            if len(a) == 2 and a[1].isdigit():
                sizes[a[0]] = int(a[1])
        for e in binaries:
            e.size = sizes.get(e.sha)
    return entries


def worktree_entries(ctx, paths):
    tracked = set()
    if paths:
        names = [p.replace("\\", "/") for p in paths]
        out = git(ctx, "ls-files", "-z", "-c", "--", *names, ok_fail=True)
        tracked = {x for x in out.decode("utf-8", "replace").split("\0") if x}
        todo = names
    else:
        out = git(ctx, "ls-files", "-z", "-c")
        tracked = {x for x in out.decode("utf-8", "replace").split("\0") if x}
        out2 = git(ctx, "ls-files", "-z", "-o", "--exclude-standard")
        todo = sorted(tracked | {x for x in out2.decode("utf-8", "replace").split("\0") if x})
    entries = []
    for rel in todo:
        full = os.path.join(ctx.root, rel)
        if not os.path.isfile(full) or os.path.islink(full):
            continue
        e = Entry(rel, "M" if rel in tracked else "A", size=os.path.getsize(full))
        if e.size <= 16 * MB:
            try:
                with open(full, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b"\0" in data[:8192]:
                e.binary = True
            else:
                text = data.decode("utf-8", "replace")
                e.lines = [(n, s.rstrip("\r")) for n, s in enumerate(text.split("\n"), 1)]
        entries.append(e)
    return entries


def read_allow_list(ctx, rel, staged_paths=(), ref="HEAD"):
    """The allow list is read from the working tree; when the commit itself changes the list, only the
    committed (HEAD, or the given ref) version counts, so a commit cannot allow-list its own additions."""
    if rel in staged_paths:
        return git(ctx, "show", ref + ":" + rel, ok_fail=True).decode("utf-8", "replace"), ref
    p = os.path.join(ctx.root, rel)
    if os.path.isfile(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            return f.read(), "worktree"
    return "", "none"


# ----------------------------------------------------------------------------- checks
def split_ext(low):
    base = low[:-5] if low.endswith(".meta") else low
    return os.path.splitext(base)[1], low.endswith(".meta")


def scan_text(ctx, path, lines, viol, where="line"):
    seen = set()
    for no, text in lines:
        if not text.strip():
            continue
        ignore = IGNORE_MARK in text
        if not ignore:
            for name, rx, ok in SECRET_PATTERNS:
                if len(text) < 8:
                    break
                for m in rx.finditer(text):
                    if ok is None or ok(m):
                        key = ("secret", no, name)
                        if key not in seen:
                            seen.add(key)
                            viol.append(Violation("secret", path, no, "looks like a %s (value not shown)" % name))
                        break
        for val in ctx.secret_values:
            if val in text:
                viol.append(Violation("secret", path, no, "contains the exact value of a local key file (value not shown)"))
                break
        if ctx.use_terms and ctx.term_any is not None:
            hits = ctx.term_hits(text)
            if hits:
                viol.append(Violation("term", path, no, "private forbidden term(s) %s (text not shown)" %
                                      ", ".join("#%d" % h for h in hits)))
        if ctx.cfg["generic"] and not ignore:
            for rx in LOCAL_PATH_RES:
                if rx.search(text):
                    viol.append(Violation("local-path", path, no, "absolute path with a user directory (use a relative path or a placeholder)"))
                    break


def check_entries(ctx, entries, diff=True):
    """diff=False: audit of whole working-tree files, where there is no baseline to compare with (no ratchet)."""
    viol = []
    limit = ctx.cfg["max_bytes"]
    for e in entries:
        path = e.path
        low = path.lower()
        ext, is_meta = split_ext(low)
        allowed = ctx.is_allowed(path)

        if any(rx.search(low) for rx in KEY_NAME_RES):
            viol.append(Violation("key-file", path, None, "file name looks like a key, token, credential or private-terms file"))

        if not is_meta:
            if ext in MODEL_DATA_EXTS or ext in CAPTURE_EXTS:
                viol.append(Violation("model-data", path, None, "weights, arrays, pickles, point clouds and splats are never committed (gated models, captures)"))
            elif ext in GATED_EXTS and GATED_TOKEN_RE.search(low):
                viol.append(Violation("gated-model", path, None, "looks like a gated body-model file (the licence forbids redistribution)"))
            if ext in MEDIA_EXTS:
                viol.append(Violation("media", path, None, "audio and video files are never committed (recordings, songs, narration)"))
            if e.is_new and (ext in IMAGE_EXTS or ext in ASSET3D_EXTS) and not allowed:
                viol.append(Violation("binary-asset", path, None, "new image or 3D-model file (may carry a likeness): add it to the allow list only with the owner's approval"))

        if ext not in CODE_EXTS and LIKENESS_NAME_RE.search(low):
            viol.append(Violation("likeness", path, None, "file name looks like portrait, texture or groom data of a person"))

        for rx, exc, why in PATH_RULES.get(ctx.profile, ()):
            if rx.search(low) and not (exc is not None and exc.search(low)):
                viol.append(Violation("private-path", path, None, why))
                break

        if e.size is not None and e.size > limit and not allowed:
            viol.append(Violation("size", path, None, "%.1f MB is above the %d MB limit (allow list only with the owner's approval)" % (e.size / MB, limit // MB)))

        if e.is_new and ctx.use_terms and ctx.term_any is not None:
            hits = ctx.term_hits(path)
            if hits:
                viol.append(Violation("term", path, None, "path contains private forbidden term(s) %s (text not shown)" %
                                      ", ".join("#%d" % h for h in hits)))

        if e.lines and not e.binary:
            scan_text(ctx, path, e.lines, viol)
    if ctx.ratchet and diff:
        ratchet_check(ctx, entries, viol)
    return viol


def has_local_path(text):
    return IGNORE_MARK not in text and any(rx.search(text) for rx in LOCAL_PATH_RES)


def ratchet_check(ctx, entries, viol):
    """Workspace profile while the remote is public: nothing may ADD lines with private terms or absolute
    user paths beyond what a file already had.  The count is per file and net (added minus removed lines),
    so editing a line that already carries a term or a path is free; a file that terms-allow.txt lists
    (take configs, world manifests: dancer data by design) is exempt from the term check.  Absolute paths
    are an error; terms are a warning unless hm.guard.terms=block, because the take names run through the
    code base and the real gate is the pre-push refusal."""
    gains = []
    for e in entries:
        if e.binary or not (e.lines or e.removed):
            continue
        if ctx.term_any is not None and not ctx.terms_allowed(e.path):
            net = sum(1 for _, s in e.lines if ctx.term_hits(s)) - sum(1 for s in e.removed if ctx.term_hits(s))
            if net > 0:
                gains.append((e.path, net))
        net = sum(1 for _, s in e.lines if has_local_path(s)) - sum(1 for s in e.removed if has_local_path(s))
        if net > 0:
            viol.append(Violation("local-path", e.path, None,
                                  "adds %d line(s) with an absolute path containing a user directory (use a relative path or a "
                                  "placeholder; the remote is public)" % net))
    if not gains:
        return
    if ctx.terms_mode == "block":
        for path, net in gains:
            viol.append(Violation("term", path, None, "adds %d line(s) with private forbidden terms (text not shown); the remote is "
                                  "public and the file is not in terms-allow.txt" % net))
        return
    shown = ", ".join("%s (+%d)" % (ctx.mask(p), n) for p, n in gains[:8])
    ctx.warnings.append("public remote: this commit adds %d line(s) with private terms in %d file(s) outside terms-allow.txt: %s%s. "
                        "The pre-push hook refuses every push while the remote is public; hm.guard.terms=block would refuse the commit."
                        % (sum(n for _, n in gains), len(gains), shown, " ..." if len(gains) > 8 else ""))


def check_message(ctx, text, label="commit message"):
    lines = [(n, s) for n, s in enumerate(text.splitlines(), 1) if not s.startswith("#")]
    viol = []
    scan_text(ctx, label, lines, viol)
    return viol


PUSH_TRAILER = ("Nothing is pushed without the owner's go-ahead.  Rules and the owner-only emergency bypass: GIT.md "
                "(docs/GIT.md in head-movement, dancecap/docs/GIT.md in the workspace).\n")


def report(ctx, viol, header, trailer=None):
    out = sys.stderr
    try:
        out.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for w in ctx.warnings:
        out.write("hm-guard: WARNING %s\n" % w)
    if not viol:
        return
    out.write("hm-guard (%s): %s - %d problem(s)\n" % (ctx.profile, header, len(viol)))
    shown = 0
    for v in viol:
        if shown >= 40:
            break
        loc = ctx.mask(v.path) if ctx.term_any is not None else v.path
        if v.line:
            loc += ":%d" % v.line
        out.write("  [%s] %s: %s\n" % (v.rule, loc, v.msg))
        shown += 1
    if len(viol) > shown:
        out.write("  ... and %d more\n" % (len(viol) - shown))
    out.write(trailer or
              "Fix it: unstage with `git restore --staged <path>`, remove or genericise the content, ignore the file in .gitignore.\n"
              "Rules and the owner-only emergency bypass: GIT.md (docs/GIT.md in head-movement, dancecap/docs/GIT.md in the workspace).\n")


# ---------------------------------------------------------------------------- pre-push
def is_ancestor(ctx, old, new):
    p = subprocess.run(["git", "merge-base", "--is-ancestor", old, new], cwd=ctx.root, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode == 0


def pre_push(ctx, remote, url, stdin_text):
    """Audit a push before anything leaves the machine (git calls the hook with the remote name and URL as
    arguments and one `<local ref> <local sha> <remote ref> <remote sha>` line per ref on stdin)."""
    viol = []
    refs = [ln.split() for ln in stdin_text.splitlines() if len(ln.split()) == 4]
    if not refs:
        return 0
    here_rel = os.path.relpath(os.path.dirname(os.path.abspath(__file__)), ctx.root).replace("\\", "/")
    allow_rel = here_rel + "/allow.txt"
    if ctx.profile == "workspace" and ctx.visibility == "public":
        viol.append(Violation(
            "push", remote, None,
            "this repository's remote is treated as PUBLIC and the repository holds dancer-specific data (take configs with names, "
            "appearance edits and footage file names; names in code, tests and docs).  Make the GitHub repository private and record it "
            "with `git config hm.visibility private`, or move the dancer data out first; the owner can override once with "
            "HM_GUARD_BYPASS=\"reason\" git push ..."))
    else:
        for local_ref, local_sha, remote_ref, remote_sha in refs:
            label = "%s -> %s" % (local_ref, remote_ref)
            if local_sha == ZERO_SHA:
                viol.append(Violation("push", label, None, "deleting a remote ref: only the repository owner does that (HM_GUARD_BYPASS)"))
                continue
            if remote_sha != ZERO_SHA and not is_ancestor(ctx, remote_sha, local_sha):
                viol.append(Violation("push", label, None,
                                      "not a fast-forward of the remote ref (history rewrite or unfetched remote commits): force-pushes are the owner's decision"))
                continue
            try:
                if remote_sha != ZERO_SHA:
                    rl = git(ctx, "rev-list", "--reverse", "--parents", local_sha, "--not", remote_sha)
                else:
                    rl = git(ctx, "rev-list", "--reverse", "--parents", local_sha, "--not", "--remotes=" + remote)
            except RuntimeError as exc:
                viol.append(Violation("push", label, None, "cannot list the commits to send: %s" % exc))
                continue
            commits = [ln.split() for ln in rl.decode("utf-8", "replace").splitlines() if ln.strip()]
            text, _src = read_allow_list(ctx, allow_rel, {allow_rel}, ref=local_sha)
            ctx.load_allow(allow_rel, text)
            tpath = here_rel + "/terms-allow.txt"
            ctx.load_terms_allow(read_allow_list(ctx, tpath, {tpath}, ref=local_sha)[0])
            for parts in commits:
                sha, parents = parts[0], parts[1:]
                short = sha[:8]
                msg = git(ctx, "log", "-1", "--format=%B", sha).decode("utf-8", "replace")
                for v in check_message(ctx, msg, "commit %s message" % short):
                    viol.append(v)
                if len(parents) > 1:
                    continue                      # a merge adds nothing its parents do not already carry
                entries = commit_entries(ctx, sha)
                for v in check_entries(ctx, entries):
                    v.path = "%s:%s" % (short, v.path)
                    viol.append(v)
    report(ctx, viol, "push to %s refused" % remote, trailer=PUSH_TRAILER)
    return 1 if viol else 0


# --------------------------------------------------------------------------- self-test
def self_test():
    here = os.path.abspath(__file__)
    tmp = tempfile.mkdtemp(prefix="hm-guard-selftest-")
    failures = []
    total = [0]
    fake_term = "zzqxterm"
    terms = os.path.join(tmp, "terms.txt")
    with open(terms, "w", encoding="utf-8") as f:
        f.write("# test terms\n%s\nsecond zzqy\n" % fake_term)
    base_env = dict(os.environ, HM_GUARD_TERMS_FILE=terms, HM_GUARD_SECRET_FILES=os.path.join(tmp, "none-*"),
                    HM_GUARD_VISIBILITY="public")
    for k in ("HM_GUARD_TERMS", "HM_GUARD_TERMS_MODE"):
        base_env.pop(k, None)
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)

    def run(*a):
        return subprocess.run(["git"] + list(a), cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    run("init", "-q")
    run("config", "user.email", "t@example.invalid")
    run("config", "user.name", "t")
    run("config", "core.autocrlf", "false")
    run("config", "commit.gpgsign", "false")
    run("config", "core.hooksPath", os.path.join(tmp, "no-hooks"))     # the throw-away repository runs no hooks of its own
    hooks = os.path.join(repo, "tools", "githooks")
    os.makedirs(hooks)
    guard_py = os.path.join(hooks, "guard.py")
    shutil.copy(here, guard_py)
    with open(os.path.join(hooks, "allow.txt"), "w") as f:
        f.write("allowed/*  # self-test\n")
    with open(os.path.join(hooks, "terms-allow.txt"), "w") as f:
        f.write("dancecap/takes/*  # self-test\n")
    run("add", "tools")
    run("commit", "-q", "-m", "init")

    def guard(profile, mode_args, extra_env=None, stdin_text=None):
        env = dict(base_env)
        env.update(extra_env or {})
        p = subprocess.run([sys.executable, guard_py, "--profile", profile] + list(mode_args), cwd=repo, env=env,
                           input=(stdin_text or "").encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return p.returncode, p.stderr.decode("utf-8", "replace")

    def verdict(label, profile, expect, rc, err):
        """expect: None = exit 0 (warnings allowed), "quiet" = exit 0 without a warning, "warn" = exit 0 with a warning,
        any other string = exit 1 and that rule id in the report."""
        total[0] += 1
        if expect is None:
            ok = rc == 0
        elif expect == "quiet":
            ok = rc == 0 and "WARNING" not in err
        elif expect == "warn":
            ok = rc == 0 and "WARNING" in err
        else:
            ok = rc == 1 and ("[%s]" % expect) in err
        if fake_term in err.lower() or "zzqy" in err.lower():
            ok = False
            err += "\n(term text echoed!)"
        print("%s %-9s %-44s expect=%s" % ("ok  " if ok else "FAIL", profile, label[:44], expect or "clean"))
        if not ok:
            failures.append((profile, label, err))

    fake_key = "sk-" + "a1B2c3D4e5" * 4
    user_dir = "C:" + chr(92) + "Users" + chr(92) + "someguy"   # built from parts so this file does not trip its own rule
    priv = {"HM_GUARD_VISIBILITY": "private"}
    cases = [
        ("unity", "notes/ok.md", "just a normal file\n", None),
        ("workspace", "notes/ok.md", "just a normal file\n", "quiet"),
        ("unity", "keys/runpod-key.txt", "x" * 40 + "\n", "key-file"),
        ("workspace", ".env", "A=1\n", "key-file"),
        ("workspace", "src/cfg.py", "KEY = '%s'\n" % fake_key, "secret"),
        ("unity", "src/cfg.cs", 'string k = "%s";\n' % fake_key, "secret"),
        ("workspace", "src/ok.py", "KEY = '%s'  # hm-guard: ignore\n" % fake_key, None),
        ("workspace", "audio/take.wav", b"RIFF0000WAVE", "media"),
        ("unity", "video/clip.mp4", b"\0\0\0\x18ftypmp42", "media"),
        ("workspace", "models/SMPLX_NEUTRAL.npz", b"PK\x03\x04", "model-data"),
        ("unity", "Assets/mano/hand.obj", "o hand\n", "gated-model"),
        ("workspace", "data/big.txt", b"a" * (5 * MB + 10), "size"),
        ("workspace", "allowed/big.txt", b"a" * (5 * MB + 10), None),
        ("unity", "Assets/ui/panel.png", b"\x89PNG\r\n\x1a\n\0\0", "binary-asset"),
        ("unity", "allowed/panel.png", b"\x89PNG\r\n\x1a\n\0\0", None),
        ("workspace", "web/portrait_ref.json", "{}\n", "likeness"),
        ("workspace", "web/groom_tool.py", "x = 1\n", "quiet"),
        ("unity", "docs/n.md", "see %s here\n" % fake_term, "term"),
        ("unity", "docs/n2.md", "see Second_ZZQY here\n", "term"),
        ("unity", "docs/%s_notes.md" % fake_term, "clean content\n", "term"),
        ("unity", "docs/p.md", "saved in " + user_dir + "\\Desktop\\x\n", "local-path"),
        ("unity", "docs/p2.md", "saved in C:\\Users\\<you>\\Desktop\\x and /home/runner/work\n", None),
        # private paths: ignored on purpose, and the ignore file alone does not stop `git add -f`
        ("unity", "Assets/StreamingAssets/cap/data.json", "{}\n", "private-path"),
        ("unity", "assets/streamingassets/cap/more.txt", "x\n", "private-path"),
        ("unity", "Assets/StreamingAssets/.gitkeep", "", None),
        ("unity", "Assets/StreamingAssets/.gitkeep.meta", "fileFormatVersion: 2\n", None),
        ("unity", "Recordings/take.json", "{}\n", "private-path"),
        ("workspace", "dancecap/work/run1/out.json", "{}\n", "private-path"),
        ("workspace", "atlas_bridge/sessions/s1/work/a.json", "{}\n", "private-path"),
        ("workspace", "dancecap/tests/test_work.py", "x = 1\n", "quiet"),
        # workspace profile: public remote (default) against private remote (the owner recorded it)
        ("workspace", "docs/n.md", "see %s here\n" % fake_term, "warn"),
        ("workspace", "docs/n.md", "see %s here\n" % fake_term, "quiet", priv),
        ("workspace", "docs/n.md", "see %s here\n" % fake_term, "quiet", {"HM_GUARD_TERMS_MODE": "off"}),
        ("workspace", "docs/n.md", "see %s here\n" % fake_term, "term", {"HM_GUARD_TERMS_MODE": "block"}),
        ("workspace", "docs/n.md", "see %s here\n" % fake_term, "term", {"HM_GUARD_TERMS": "1"}),
        ("workspace", "dancecap/takes/t.toml", "name = '%s'\n" % fake_term, "quiet"),
        ("workspace", "docs/p.md", "saved in " + user_dir + "\\Desktop\\x\n", "local-path"),
        ("workspace", "docs/p.md", "saved in " + user_dir + "\\Desktop\\x\n", None, priv),
        ("workspace", "docs/p3.md", "saved in " + user_dir + "\\Desktop\\x  hm-guard: ignore\n", "quiet"),
    ]
    for case in cases:
        profile, rel, content, expect = case[:4]
        extra = case[4] if len(case) > 4 else None
        full = os.path.join(repo, rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as f:
            f.write(content if isinstance(content, bytes) else content.encode())
        run("add", "-f", "--", rel)
        rc, err = guard(profile, ["--staged"], extra)
        verdict(rel + (" " + "/".join(extra.values()) if extra else ""), profile, expect, rc, err)
        run("rm", "-q", "--cached", "-f", "--", rel)
        try:
            os.remove(full)
        except OSError:
            pass

    # ratchet: a file that already carries a term / a user path may keep it, but not gain more
    for rel, content in (("docs/base.md", "keep %s line\nplain line\n" % fake_term),
                         ("src/old.py", "P = '" + user_dir + "\\x'\nplain = 1\n")):
        os.makedirs(os.path.dirname(os.path.join(repo, rel)), exist_ok=True)
        with open(os.path.join(repo, rel), "w") as f:
            f.write(content)
        run("add", "--", rel)
    run("commit", "-q", "-m", "baseline with a term and a path")
    ratchet_cases = [
        ("docs/base.md", "edited %s line\nplain line\n" % fake_term, "quiet", None),
        ("docs/base.md", "plain line\nmoved %s line\n" % fake_term, "quiet", None),
        ("docs/base.md", "keep %s line\nplain line\nnew %s line\n" % (fake_term, fake_term), "warn", None),
        ("docs/base.md", "keep %s line\nplain line\nnew %s line\n" % (fake_term, fake_term), "term", {"HM_GUARD_TERMS_MODE": "block"}),
        ("docs/base.md", "keep %s line\nplain line\nnew %s line\n" % (fake_term, fake_term), "quiet", priv),
        ("src/old.py", "P = '" + user_dir + "\\y'\nplain = 1\n", "quiet", None),
        ("src/old.py", "P = '" + user_dir + "\\x'\nQ = '" + user_dir + "\\z'\n", "local-path", None),
        ("src/old.py", "P = '" + user_dir + "\\x'\nQ = '" + user_dir + "\\z'\n", None, priv),
    ]
    for rel, content, expect, extra in ratchet_cases:
        with open(os.path.join(repo, rel), "w") as f:
            f.write(content)
        run("add", "--", rel)
        rc, err = guard("workspace", ["--staged"], extra)
        verdict("ratchet " + rel + (" private" if extra else ""), "workspace", expect, rc, err)
        run("reset", "-q", "--hard")

    # commit-message mode
    mf = os.path.join(tmp, "msg.txt")
    for text, expect, profile in (("fix thing\n\nmentions %s\n" % fake_term, "term", "unity"), ("fix thing\n# comment %s\n" % fake_term, None, "unity"),
                                  ("key %s\n" % fake_key, "secret", "workspace"), ("plain message\n", None, "workspace")):
        with open(mf, "w") as f:
            f.write(text)
        rc, err = guard(profile, ["--commit-msg", mf])
        verdict("commit-msg", profile, expect, rc, err)

    # pre-push: stdin lines are `<local ref> <local sha> <remote ref> <remote sha>`
    def head():
        return run("rev-parse", "HEAD").stdout.decode().strip()

    def commit(rel, content, msg):
        if rel:
            full = os.path.join(repo, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w") as f:
                f.write(content)
            run("add", "-f", "--", rel)
        run("commit", "-q", "--allow-empty", "-m", msg)
        return head()

    base = head()
    c_ok = commit("notes/fine.md", "fine\n", "add a fine note")
    c_term = commit("notes/leak.md", "leaks %s\n" % fake_term, "add a note")
    c_msg = commit(None, "", "mentions %s" % fake_term)
    c_key = commit("notes/key.md", "k = '%s'\n" % fake_key, "add a note with a key")
    c_path = commit("notes/where.md", "at " + user_dir + "\\Desktop\n", "add a note with a path")
    c_shot = commit("Assets/StreamingAssets/cap/d.json", "{}\n", "force-added capture data")
    push_cases = [
        ("clean commit", "unity", c_ok, base, None, None),
        ("private term in a file", "unity", c_term, c_ok, "term", None),
        ("private term in a message", "unity", c_msg, c_term, "term", None),
        ("absolute user path", "unity", c_path, c_key, "local-path", None),
        ("capture data added with -f", "unity", c_shot, c_path, "private-path", None),
        ("whole history of a new branch", "unity", c_term, ZERO_SHA, "term", None),
        ("deleting a remote branch", "unity", ZERO_SHA, base, "push", None),
        ("non-fast-forward", "unity", base, c_ok, "push", None),
        ("workspace, remote treated as public", "workspace", c_ok, base, "push", None),
        ("workspace, remote private, clean", "workspace", c_ok, base, None, priv),
        ("workspace, remote private, term (data by design)", "workspace", c_term, c_ok, None, priv),
        ("workspace, remote private, secret", "workspace", c_key, c_msg, "secret", priv),
        ("workspace, remote private, capture-data path (unity rule only)", "workspace", c_shot, c_path, None, priv),
    ]
    for label, profile, local_sha, remote_sha, expect, extra in push_cases:
        stdin_text = "refs/heads/main %s refs/heads/main %s\n" % (local_sha, remote_sha)
        rc, err = guard(profile, ["--pre-push", "origin", "https://example.invalid/r.git"], extra, stdin_text)
        verdict("pre-push: " + label, profile, expect, rc, err)
    rc, err = guard("unity", ["--pre-push", "origin", "https://example.invalid/r.git"], None, "")
    verdict("pre-push: nothing to push", "unity", None, rc, err)

    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print("\nSELF-TEST FAILED (%d of %d)" % (len(failures), total[0]))
        for prof, rel, err in failures:
            print("--", prof, rel)
            print(err)
        return 1
    print("\nSELF-TEST PASSED (%d cases)" % total[0])
    return 0


# ------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="hm-guard commit guard (see module docstring)")
    ap.add_argument("--profile", choices=sorted(PROFILES))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--staged", action="store_true")
    g.add_argument("--commit-msg", metavar="FILE")
    g.add_argument("--pre-push", nargs=2, metavar=("REMOTE", "URL"))
    g.add_argument("--worktree", nargs="*", metavar="PATH")
    g.add_argument("--self-test", action="store_true")
    g.add_argument("--version", action="store_true")
    ap.add_argument("--stdin-file", metavar="FILE", help="read the pre-push ref lines from FILE instead of stdin")
    a = ap.parse_args(argv)

    if a.version:
        with open(os.path.abspath(__file__), "rb") as f:
            print("hm-guard %s sha256=%s" % (VERSION, hashlib.sha256(f.read().replace(b"\r\n", b"\n")).hexdigest()[:16]))
        return 0
    if a.self_test:
        return self_test()
    if not a.profile:
        ap.error("--profile is required")

    ctx = Ctx(a.profile, repo_root())
    here_rel = os.path.relpath(os.path.dirname(os.path.abspath(__file__)), ctx.root).replace("\\", "/")
    allow_rel = here_rel + "/allow.txt"

    if a.commit_msg:
        with open(a.commit_msg, encoding="utf-8", errors="replace") as f:
            viol = check_message(ctx, f.read())
        report(ctx, viol, "commit message rejected")
        return 1 if viol else 0

    if a.pre_push:
        if a.stdin_file:
            with open(a.stdin_file, encoding="utf-8", errors="replace") as f:
                stdin_text = f.read()
        else:
            stdin_text = sys.stdin.read()
        return pre_push(ctx, a.pre_push[0], a.pre_push[1], stdin_text)

    if a.staged:
        entries = staged_entries(ctx)
        if not entries:
            report(ctx, [], "")
            return 0
        staged_paths = {e.path for e in entries}
        text, _src = read_allow_list(ctx, allow_rel, staged_paths)
        ctx.load_allow(allow_rel, text)
        terms_rel = here_rel + "/terms-allow.txt"
        ctx.load_terms_allow(read_allow_list(ctx, terms_rel, staged_paths)[0])
        if allow_rel in staged_paths:
            ctx.warnings.append("the allow list is part of this commit; the new entries apply from the NEXT commit")
        viol = check_entries(ctx, entries)
        report(ctx, viol, "commit blocked")
        return 1 if viol else 0

    text, _src = read_allow_list(ctx, allow_rel)
    ctx.load_allow(allow_rel, text)
    entries = worktree_entries(ctx, a.worktree)
    viol = check_entries(ctx, entries, diff=False)
    report(ctx, viol, "audit of %d file(s)" % len(entries))
    print("hm-guard audit (%s): %d file(s) checked, %d problem(s)" % (ctx.profile, len(entries), len(viol)))
    return 1 if viol else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # fail closed: a broken guard must not wave a commit through silently
        sys.stderr.write("hm-guard: internal error: %s\n" % exc)
        sys.exit(2)
