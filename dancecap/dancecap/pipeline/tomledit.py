"""Comment-preserving edits of known scalar keys in a take TOML (PIPELINE_MONITOR 7.4, "owned keys").

Only `key = value` lines of known keys inside a named table are rewritten (a trailing comment is kept); a missing key
is inserted after the last key line of its table; a missing table is appended. Every other byte stays as it was.
The result is parsed with tomllib and compared with the intent before anything is written; writes check the etag
(sha1 of the bytes), back the old file up under work/<take>/monitor/toml_backups/ and replace it atomically.
"""
from __future__ import annotations

import json
import re
import time
import tomllib

from . import paths
from .util import sha1_bytes, write_text_atomic

_HDR = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")


class StaleEtag(RuntimeError):
    def __init__(self, etag: str):
        super().__init__("the take TOML changed since it was read")
        self.etag = etag


def value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {value(x)}" for k, x in v.items()) + " }"
    raise TypeError(type(v).__name__)


def _norm_header(h: str) -> str:
    return ".".join(p.strip().strip('"').strip("'") for p in h.split("."))


def _table_span(lines: list[str], table: str | None) -> tuple[int, int] | None:
    """[start, end) line span of a table body (top level: from 0 to the first header)."""
    if table is None:
        end = next((i for i, l in enumerate(lines) if _HDR.match(l.rstrip("\r\n"))), len(lines))
        return 0, end
    for i, l in enumerate(lines):
        m = _HDR.match(l.rstrip("\r\n"))
        if m and not l.lstrip().startswith("[[") and _norm_header(m.group(1)) == table:
            j = i + 1
            while j < len(lines) and not _HDR.match(lines[j].rstrip("\r\n")):
                j += 1
            return i + 1, j
    return None


def set_keys(text: str, table: str | None, kv: dict) -> str:
    lines = text.splitlines(keepends=True)
    span = _table_span(lines, table)
    if span is None:
        add = ([] if not lines or lines[-1].endswith("\n") else ["\n"]) + ["\n", f"[{table}]\n"]
        add += [f"{k} = {value(v)}\n" for k, v in kv.items() if v is not None]
        return "".join(lines + add)
    a, b = span
    rest = dict(kv)
    for i in range(a, b):
        m = re.match(r"^(\s*)([A-Za-z0-9_-]+)(\s*=\s*)(.*)$", lines[i].rstrip("\r\n"))
        if not m or m.group(2) not in rest:
            continue
        k = m.group(2)
        v = rest.pop(k)
        tail = ""
        cm = re.search(r"\s+#[^\"']*$", m.group(4))
        if cm and not m.group(4).strip().startswith(("{", "[")):
            tail = cm.group(0)
        if v is None:
            lines[i] = ""
        else:
            lines[i] = f"{m.group(1)}{k}{m.group(3)}{value(v)}{tail}\n"
    if rest:
        last = a - 1
        for i in range(a, b):
            if re.match(r"^\s*[A-Za-z0-9_\"'-]+\s*=", lines[i]):
                last = i
        ins = [f"{k} = {value(v)}\n" for k, v in rest.items() if v is not None]
        lines[last + 1:last + 1] = ins
    return "".join(lines)


def write(take: str, edits: list[tuple[str | None, dict]], if_match: str | None, check=None) -> str:
    """Apply [(table, {key: value})] edits; returns the new etag. check(parsed) may raise to refuse the result."""
    p = paths.take_toml(take)
    b = p.read_bytes()
    cur = sha1_bytes(b)
    if if_match is not None and if_match.strip('"') != cur:
        raise StaleEtag(cur)
    text = b.decode("utf-8-sig")
    new = text
    for table, kv in edits:
        new = set_keys(new, table, kv)
    parsed = tomllib.loads(new)
    for table, kv in edits:
        node = parsed
        for part in (table.split(".") if table else []):
            node = node.get(part, {})
        for k, v in kv.items():
            if v is not None and node.get(k) != v:
                raise ValueError(f"write check failed for {table}.{k}")
    if check:
        check(parsed)
    bdir = paths.take_work(take) / "monitor" / "toml_backups"
    bdir.mkdir(parents=True, exist_ok=True)
    (bdir / f"{time.strftime('%Y%m%d-%H%M%S')}-{take}.toml").write_bytes(b)
    write_text_atomic(p, new)
    return sha1_bytes(new.encode("utf-8"))
