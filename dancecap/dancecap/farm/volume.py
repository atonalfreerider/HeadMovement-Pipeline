"""The hm-farm network volume: create / describe / sync up / sync down / ls, with hash-diffed resumable transfers.

The volume is reachable only through a pod that mounts it (Runpod's S3 API needs console-created keys, not available to
this tooling), so every operation takes a `Link` = an attached pod (a cheap CPU gateway pod when no GPU pod is up):

    from dancecap.farm import pods, volume
    link = volume.Link(pods.attach(pod_id))
    link.push_file(Path("weights/x.ckpt"), "weights/mamma/ma_2d/x.ckpt")      # resumable, skipped when sha256 matches
    volume.sync_code(link)                                                     # code bundle at a content key
    link.pull_file("chunks/c01/out/result.tar", Path("work/farm/c01/result.tar"), size=.., sha256=..)
    link.ls("chunks/c01/out")

Measured on 2026-10-08 (EU-RO-1 MooseFS volume): 1.2 GB/s sequential write, 480 MB/s read, but ~24 ms per small file
created and ~12 ms per small file read. Consequences baked in here: transfers are single big files (tar a tree before
pushing it, `pack_tree`), the environment is a tar unpacked on the pod's local disk, and the worker keeps its scratch
space on the local disk.

Layout under /workspace/hm (FARM.md section 4): env/ (env-<id>.tar + CURRENT), wheels/, weights/, models/, code/
(<key>.tar.gz + CURRENT), worlds/<world_id>/, src/, queue/, chunks/<id>/{in,out,log,claim.lock,heartbeat,done.json},
pods/<pod id>/, bin/, debs/, logs/.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .. import remote, remote_jobs
from ..take import PACKAGE_ROOT, WORKSPACE

ROOT = "/workspace/hm"
LAYOUT = ("env", "wheels", "weights", "models", "code", "worlds", "src", "queue", "chunks", "pods", "bin", "debs", "logs",
          "tmp")
SHA_CACHE = PACKAGE_ROOT / "work" / "farm" / "sha_cache.json"
BOOTSTRAP_DIR = Path(__file__).parent / "bootstrap"
# compressing these on the ssh link only burns CPU
INCOMPRESSIBLE = (".pt", ".pth", ".ckpt", ".npz", ".gz", ".zst", ".mp4", ".mov", ".jpg", ".jpeg", ".png", ".whl", ".tar.gz",
                  ".safetensors", ".bin")
SMALL = 4 << 20       # files below this ride in tar batches
SHARED_PREFIXES = ("worlds/", "src/", "weights/", "models/", "chunks/")   # = worker.SHARED_PREFIXES
STRIPE_MIN = 24 << 20  # files from this size are striped over several ssh streams
MAX_SLOTS = int(os.environ.get("HM_PUSH_SLOTS", "8"))   # concurrent ssh data connections to one pod (sshd's MaxStartups is 10)
_SLOTS = threading.BoundedSemaphore(MAX_SLOTS)
DEFAULT_STREAMS = int(os.environ.get("HM_PUSH_STREAMS", "6"))
BATCH_BYTES = 16 << 20
CHUNK = 1 << 20


# ------------------------------------------------------------------------------------------------ hashing


def _load_cache() -> dict:
    try:
        return json.loads(SHA_CACHE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def sha256_file(path: Path, use_cache: bool = True) -> str:
    """sha256 of a local file, cached by (path, size, mtime_ns) in work/farm/sha_cache.json."""
    st = path.stat()
    key = f"{st.st_size}:{st.st_mtime_ns}"
    cache = _load_cache() if use_cache else {}
    hit = cache.get(str(path))
    if hit and hit["key"] == key:
        return hit["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    digest = h.hexdigest()
    if use_cache:
        cache[str(path)] = {"key": key, "sha256": digest}
        try:
            SHA_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = SHA_CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(cache))
            os.replace(tmp, SHA_CACHE)
        except OSError:
            pass
    return digest


def _hash_range(path: Path, start: int, length: int) -> str:
    """sha256 of bytes [start, start+length) of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(start)
        left = length
        while left > 0:
            b = fh.read(min(left, 1 << 22))
            if not b:
                break
            h.update(b)
            left -= len(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ link to the volume


@dataclass
class Link:
    """A pod that mounts the volume, plus the shell helpers the transfers need. Safe to share between threads."""
    pod: remote.Pod
    root: str = ROOT

    # -- shell
    def sh(self, cmd: str, timeout: float = 300) -> str:
        return self.pod.output(cmd, timeout=timeout)

    def abs(self, rel: str) -> str:
        rel = rel.replace("\\", "/").lstrip("/")
        if ".." in PurePosixPath(rel).parts:
            raise ValueError(f"bad volume path {rel}")
        return f"{self.root}/{rel}"

    # -- reads
    def ls(self, rel: str = "", sizes: bool = True) -> list[dict]:
        out = self.sh(f"cd {shlex.quote(self.abs(rel))} 2>/dev/null && find . -maxdepth 1 -mindepth 1 -printf '%y\\t%s\\t%T@\\t%f\\n' | sort -k4", 120)
        rows = []
        for line in out.splitlines():
            t, size, mtime, name = line.split("\t", 3)
            rows.append({"name": name, "type": "dir" if t == "d" else "file", "bytes": int(size), "mtime": float(mtime)})
        return rows

    def exists(self, rel: str) -> bool:
        return self.sh(f"[ -e {shlex.quote(self.abs(rel))} ] && echo 1 || echo 0", 60).strip() == "1"

    def read_text(self, rel: str, default: str | None = None) -> str | None:
        try:
            return self.sh(f"cat {shlex.quote(self.abs(rel))}", 120)
        except RuntimeError:
            return default

    def read_json(self, rel: str, default=None):
        t = self.read_text(rel)
        try:
            return json.loads(t) if t else default
        except json.JSONDecodeError:
            return default

    def write_text(self, rel: str, text: str) -> None:
        """Atomic: temp file then rename (readers never see a partial file)."""
        dst = self.abs(rel)
        self._send(f"mkdir -p {shlex.quote(os.path.dirname(dst))} && cat > {shlex.quote(dst + '.tmp')} && mv {shlex.quote(dst + '.tmp')} {shlex.quote(dst)}",
                   text.encode("utf-8"), compress=False)

    def state(self, rels: list[str], hash_equal_size: dict[str, int] | None = None) -> dict[str, tuple[int, str | None]]:
        """{rel: (size, sha256 or None)} for the files that exist. sha256 is computed only for the rels in
        `hash_equal_size` whose size equals the given local size (everything else is not worth reading)."""
        if not rels:
            return {}
        listing = "\n".join(self.abs(r) for r in rels)
        out = self.pod.output("while IFS= read -r f; do [ -f \"$f\" ] && stat -c '%s\t%n' \"$f\"; done <<'HM_LIST'\n" + listing + "\nHM_LIST\ntrue\n", timeout=300)
        got: dict[str, tuple[int, str | None]] = {}
        for line in out.splitlines():
            size, path = line.split("\t", 1)
            got[path[len(self.root) + 1:]] = (int(size), None)
        need = [r for r, (s, _) in got.items() if hash_equal_size and hash_equal_size.get(r) == s]
        if need:
            lst = "\n".join(self.abs(r) for r in need)
            out = self.pod.output("xargs -d '\\n' -r -P4 -n8 sha256sum <<'HM_LIST'\n" + lst + "\nHM_LIST\ntrue\n", timeout=1800)
            for line in out.splitlines():
                digest, path = line.rstrip("\r").split(None, 1)   # "<sha>  <path>" (GNU) or "<sha> *<path>" (MSYS)
                path = path[1:] if path.startswith("*") else path
                rel = path[len(self.root) + 1:]
                got[rel] = (got[rel][0], digest)
        return got

    # -- transport (ssh stdin/stdout streams: resumable by byte offset)
    def _ssh(self, remote_cmd: str, compress: bool) -> list[str]:
        base = self.pod._ssh_base()
        if compress:
            base = base[:1] + ["-o", "Compression=yes"] + base[1:]
        return base + [f"bash -c {shlex.quote(remote_cmd)}"]

    def _send(self, remote_cmd: str, data: bytes, compress: bool) -> None:
        with _SLOTS:
            p = subprocess.run(self._ssh(remote_cmd, compress), input=data, capture_output=True, timeout=1800)
        if p.returncode != 0:
            raise RuntimeError(f"ssh send failed ({p.returncode}): {p.stderr.decode(errors='replace')[-300:]}")

    def _upload_range(self, local: Path, start: int, length: int, part: str, compress: bool, report=None) -> int:
        """Send bytes [start, start+length) of `local` into the remote file `part`, resuming at the bytes already there when
        they are the expected prefix (a stale part of a changed file is detected by hash and restarted). Returns bytes sent."""
        q = shlex.quote
        have = int(self.sh(f"mkdir -p {q(os.path.dirname(part))}; [ -f {q(part)} ] && stat -c %s {q(part)} || echo 0", 60).strip() or 0)
        off = have if 0 < have <= length else 0
        if off:
            remote_h = self.sh(f"head -c {off} {q(part)} | sha256sum | cut -d' ' -f1", 1800).strip()
            if remote_h != _hash_range(local, start, off):
                off = 0
        if off == length:
            return 0
        cmd = f"cat > {q(part)}" if off == 0 else f"truncate -s {off} {q(part)} && cat >> {q(part)}"
        _SLOTS.acquire()
        proc = subprocess.Popen(self._ssh(cmd, compress), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        sent = 0
        try:
            with open(local, "rb") as fh:
                fh.seek(start + off)
                left = length - off
                while left > 0:
                    blk = fh.read(min(CHUNK, left))
                    if not blk:
                        break
                    proc.stdin.write(blk)
                    left -= len(blk)
                    sent += len(blk)
                    if report:
                        report(len(blk))
            proc.stdin.close()
            rc = proc.wait()
        finally:
            if proc.poll() is None:
                proc.kill()
            _SLOTS.release()
        if rc != 0:
            raise RuntimeError(f"ssh rc={rc}: {proc.stderr.read().decode(errors='replace')[-200:]}")
        return sent

    def push_file(self, local: Path, rel: str, *, force: bool = False, progress=None, retries: int = 6,
                  streams: int | None = None) -> dict:
        """Upload one file to <root>/<rel>, resumable and hash-diffed. Returns {rel, bytes, sent, seconds, skipped, streams}.

        Files of at least STRIPE_MIN bytes are striped over `streams` parallel ssh connections (default HM_PUSH_STREAMS=6):
        on the shared home uplink every connection gets about the same share (measured 2026-10-08: one stream 0.08-0.13 MB/s,
        four streams 0.42 MB/s in aggregate), so N streams move the file ~N times faster. Each stripe resumes on its own and is
        assembled on the pod; the whole-file sha256 is checked before the atomic rename."""
        local = Path(local)
        size = local.stat().st_size
        digest = sha256_file(local)
        dst = self.abs(rel)
        t0 = time.time()
        if not force:
            st = self.state([rel], {rel: size}).get(rel)
            if st and st[0] == size and st[1] == digest:
                return {"rel": rel, "bytes": size, "sent": 0, "seconds": 0.0, "skipped": True, "streams": 0}
        n = max(1, int(streams if streams is not None else DEFAULT_STREAMS)) if size >= STRIPE_MIN else 1
        seg = -(-size // n // CHUNK) * CHUNK if n > 1 else size
        ranges = [(i * seg, min(seg, size - i * seg)) for i in range(n) if i * seg < size] or [(0, 0)]
        n = len(ranges)
        compress = not rel.endswith(INCOMPRESSIBLE)
        q = shlex.quote
        lock = threading.Lock()
        done = [0]

        def report(k: int) -> None:
            with lock:
                done[0] += k
                if progress:
                    progress(rel, done[0], size)

        for attempt in range(retries):
            try:
                parts = [f"{dst}.part" if n == 1 else f"{dst}.part{n}.{i}" for i in range(n)]
                if n == 1:
                    sent = self._upload_range(local, 0, size, parts[0], compress, report)
                else:
                    with ThreadPoolExecutor(max_workers=n) as ex:
                        sent = sum(ex.map(lambda a: self._upload_range(local, a[1][0], a[1][1], parts[a[0]], compress, report),
                                          enumerate(ranges)))
                    self.sh(f"cat {' '.join(q(x) for x in parts)} > {q(dst + '.part')} && rm -f {' '.join(q(x) for x in parts)}", 1800)
                got = self.sh(f"sha256sum {q(dst + '.part')} | cut -d' ' -f1", 1800).strip()
                if got != digest:
                    self.sh(f"rm -f {q(dst + '.part')} {' '.join(q(x) for x in parts)}", 60)
                    raise RuntimeError(f"sha256 mismatch after upload of {rel}: {got[:12]} != {digest[:12]}; restarting")
                self.sh(f"mv {q(dst + '.part')} {q(dst)}", 120)
                return {"rel": rel, "bytes": size, "sent": sent, "seconds": round(time.time() - t0, 1), "skipped": False,
                        "streams": n}
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                if attempt == retries - 1:
                    raise
                print(f"[volume] push {rel}: {e}; retry {attempt + 2}/{retries} in 5 s", flush=True)
                time.sleep(5)
        raise RuntimeError("unreachable")

    def push_batch(self, files: list[tuple[Path, str]]) -> int:
        """Many small files in one tar stream (one ssh call instead of one per file). Returns bytes sent."""
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w", format=tarfile.GNU_FORMAT) as tar:
            for local, rel in files:
                ti = tarfile.TarInfo(rel.replace("\\", "/").lstrip("/"))
                ti.size = Path(local).stat().st_size
                ti.mtime = int(Path(local).stat().st_mtime)
                ti.mode = 0o644
                with open(local, "rb") as fh:
                    tar.addfile(ti, fh)
        data = buf.getvalue()
        self._send(f"mkdir -p {shlex.quote(self.root)} && tar -xf - -C {shlex.quote(self.root)}", data, compress=True)
        return len(data)

    def push_files(self, items: list[tuple[Path, str]], *, force: bool = False, progress=None, workers: int = 4) -> dict:
        """Diff `items` ((local file, volume-relative path)) against the volume and send only what differs: small files in
        tar batches of ~16 MB, large ones striped and resumable; up to `workers` of them at the same time (the ssh data
        connections are capped by MAX_SLOTS overall). Returns a summary."""
        t0 = time.time()
        local = {rel: (Path(p), Path(p).stat().st_size) for p, rel in items}
        remote_state = {} if force else self.state(list(local), {r: s for r, (_, s) in local.items()})
        todo, skipped = [], 0
        for rel, (p, size) in local.items():
            st = remote_state.get(rel)
            if st and st[0] == size and st[1] == sha256_file(p):
                skipped += 1
            else:
                todo.append((p, rel, size))
        small = [(p, rel) for p, rel, s in todo if s < SMALL]
        big = [(p, rel, s) for p, rel, s in todo if s >= SMALL]
        batches, batch, acc = [], [], 0
        for p, rel in small:
            batch.append((p, rel))
            acc += Path(p).stat().st_size
            if acc >= BATCH_BYTES:
                batches.append(batch)
                batch, acc = [], 0
        if batch:
            batches.append(batch)
        jobs = [lambda b=b: self.push_batch(b) for b in batches]
        jobs += [lambda p=p, rel=rel: self.push_file(p, rel, force=True, progress=progress)["sent"] for p, rel, _ in big]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            sent = sum(ex.map(lambda f: f(), jobs))
        if small:  # verify the batches landed intact
            vs = self.state([r for _, r in small], {r: local[r][1] for _, r in small})
            bad = [r for _, r in small if r not in vs or vs[r][0] != local[r][1]
                   or (vs[r][1] and vs[r][1] != sha256_file(local[r][0]))]
            if bad:
                raise RuntimeError(f"{len(bad)} small files not intact after upload, first {bad[0]}")
        return {"files": len(local), "skipped": skipped, "sent_files": len(todo), "sent_bytes": sent,
                "seconds": round(time.time() - t0, 1)}

    def push_tree(self, local_dir: Path, rel_dir: str, *, exclude: tuple[str, ...] = ("__pycache__", ".git"),
                  force: bool = False, progress=None) -> dict:
        items = []
        for f in sorted(Path(local_dir).rglob("*")):
            if f.is_file() and not any(x in f.parts for x in exclude):
                items.append((f, f"{rel_dir.rstrip('/')}/{f.relative_to(local_dir).as_posix()}"))
        return self.push_files(items, force=force, progress=progress)

    def pull_file(self, rel: str, local: Path, *, size: int | None = None, sha256: str | None = None, retries: int = 6,
                  progress=None) -> dict:
        """Download <root>/<rel> to `local`, resumable (byte offset of the .part) and verified. Returns {bytes, seconds}."""
        local = Path(local)
        local.parent.mkdir(parents=True, exist_ok=True)
        src = self.abs(rel)
        t0 = time.time()
        if size is None or sha256 is None:
            out = self.sh(f"stat -c %s {shlex.quote(src)}; sha256sum {shlex.quote(src)} | cut -d' ' -f1", 1800).split()
            size, sha256 = int(out[0]), out[1]
        if local.exists() and local.stat().st_size == size and sha256_file(local, use_cache=False) == sha256:
            return {"rel": rel, "bytes": size, "seconds": 0.0, "skipped": True}
        part = local.with_name(local.name + ".part")
        compress = not rel.endswith(INCOMPRESSIBLE)
        for attempt in range(retries):
            try:
                off = part.stat().st_size if part.exists() else 0
                if off > size:
                    part.unlink()
                    off = 0
                if off < size:
                    cmd = f"tail -c +{off + 1} {shlex.quote(src)}"
                    proc = subprocess.Popen(self._ssh(cmd, compress), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    try:
                        with open(part, "ab") as fh:
                            pos = off
                            while True:
                                blk = proc.stdout.read(CHUNK)
                                if not blk:
                                    break
                                fh.write(blk)
                                pos += len(blk)
                                if progress:
                                    progress(rel, pos, size)
                        rc = proc.wait()
                    finally:
                        if proc.poll() is None:
                            proc.kill()
                    if rc != 0 and part.stat().st_size < size:
                        raise RuntimeError(f"ssh rc={rc}: {proc.stderr.read().decode(errors='replace')[-200:]}")
                if part.stat().st_size != size:
                    raise RuntimeError(f"size {part.stat().st_size} != {size}")
                got = sha256_file(part, use_cache=False)
                if got != sha256:
                    part.unlink()
                    raise RuntimeError(f"sha256 mismatch after download of {rel}: {got[:12]} != {sha256[:12]}; restarting")
                os.replace(part, local)
                return {"rel": rel, "bytes": size, "seconds": round(time.time() - t0, 1), "skipped": False}
            except (RuntimeError, OSError, subprocess.SubprocessError) as e:
                if attempt == retries - 1:
                    raise
                wait = min(60, 5 * 2 ** attempt)
                print(f"[volume] pull {rel}: {e}; retry {attempt + 2}/{retries} in {wait} s", flush=True)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    def disk(self) -> dict:
        """Bytes used under <root> and per top-level directory (du; sequential, fine on a few big files). `df` on the volume
        shows Runpod's whole storage pool (petabytes), not this volume's quota, so only du is meaningful."""
        out = self.sh(f"cd {shlex.quote(self.root)} && du -B1 --max-depth=1 . 2>/dev/null | sort -k2", 900).splitlines()
        dirs = {}
        for line in out:
            parts = line.split("	", 1)
            if len(parts) == 2 and parts[0].isdigit():
                dirs[parts[1][2:] if parts[1].startswith("./") else "."] = int(parts[0])
        return {"hm_bytes": dirs.get(".", 0), "dirs": {k: v for k, v in dirs.items() if k != "."}}


# ------------------------------------------------------------------------------------------------ code bundle


CODE_EXCLUDE = ("__pycache__", ".pyc", ".git", ".pytest_cache", "node_modules", ".ipynb_checkpoints", "/work/", "/weights/")


def code_files() -> list[tuple[Path, str]]:
    """(local file, archive name relative to the workspace) of the code a pod needs: dancecap, takes, remote scripts,
    CONTRACTS.md and the patched MAMMA checkout (minus docs/gui/data/tests...)."""
    items: list[tuple[Path, str]] = [(PACKAGE_ROOT / "dancecap", "dancecap/dancecap"), (PACKAGE_ROOT / "takes", "dancecap/takes"),
                                     (PACKAGE_ROOT / "remote", "dancecap/remote"), (PACKAGE_ROOT / "CONTRACTS.md", "dancecap/CONTRACTS.md")]
    root = remote_jobs.env.mamma_root()
    for p in sorted(root.iterdir()) if root.exists() else []:
        if p.name in remote_jobs.MAMMA_SKIP or p.name.startswith("."):
            continue
        items.append((p, f"mamma-turing/{p.name}"))
    out = []
    for local, arc in items:
        if local.is_file():
            out.append((local, arc))
            continue
        for f in sorted(local.rglob("*")):
            a = f"{arc}/{f.relative_to(local).as_posix()}"
            if f.is_file() and not any(x in "/" + a if x.startswith("/") else x in a for x in CODE_EXCLUDE):
                out.append((f, a))
    return out


def _git(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def code_info() -> dict:
    """{key, git_sha, dirty, mamma_sha, n_files, bytes}: key = first 12 hex of the sha256 over (path, content sha256)."""
    files = code_files()
    h = hashlib.sha256()
    total = 0
    for f, arc in files:
        h.update(f"{arc}\0".encode())
        h.update(sha256_file(f).encode())
        total += f.stat().st_size
    dirty = bool(_git(["status", "--porcelain", "--", "dancecap"], WORKSPACE))
    return {"key": h.hexdigest()[:12], "git_sha": _git(["rev-parse", "HEAD"], WORKSPACE), "dirty": dirty,
            "mamma_sha": _git(["rev-parse", "HEAD"], remote_jobs.env.mamma_root()), "n_files": len(files), "bytes": total}


def build_code_bundle(out_dir: Path | None = None) -> tuple[Path, dict]:
    """code-<key>.tar.gz in work/farm/bundles (content-keyed, reused). Returns (path, info incl. sha256)."""
    info = code_info()
    out_dir = out_dir or PACKAGE_ROOT / "work" / "farm" / "bundles"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{info['key']}.tar.gz"
    if not out.exists():
        tmp = out.with_suffix(".part")
        with tarfile.open(tmp, "w:gz", compresslevel=6, format=tarfile.GNU_FORMAT) as tar:
            for f, arc in code_files():
                ti = tarfile.TarInfo(arc)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = "root"
                ti.mtime = int(f.stat().st_mtime)
                if arc.endswith(".sh"):
                    data = f.read_bytes().replace(b"\r\n", b"\n")
                    ti.mode, ti.size = 0o755, len(data)
                    tar.addfile(ti, io.BytesIO(data))
                else:
                    ti.mode, ti.size = 0o644, f.stat().st_size
                    with open(f, "rb") as fh:
                        tar.addfile(ti, fh)
        os.replace(tmp, out)
        for old in sorted(out_dir.glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[6:]:
            old.unlink(missing_ok=True)
    info.update({"path": str(out), "bundle_bytes": out.stat().st_size, "bundle_sha256": sha256_file(out)})
    return out, info


def sync_code(link: Link, *, make_current: bool = True) -> dict:
    """Upload the current code tree as code/<key>.tar.gz (skipped when present), point code/CURRENT at it, refresh
    bin/ (pod scripts) and write code/<key>.json. Returns the code info (git sha, dirty flag, bundle sha256)."""
    bundle, info = build_code_bundle()
    r = link.push_file(bundle, f"code/{info['key']}.tar.gz")
    scripts = [(BOOTSTRAP_DIR / n, f"bin/{n}") for n in ("hm_killswitch.sh", "pod_entry.sh", "pod_prepare.sh", "hm_env.sh", "manifest.py", "import_check.py")]
    tmp = PACKAGE_ROOT / "work" / "farm" / "bin"
    tmp.mkdir(parents=True, exist_ok=True)
    staged = []
    for src, rel in scripts:
        dst = tmp / Path(rel).name
        dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
        staged.append((dst, rel))
    link.push_files(staged)
    link.sh(f"chmod +x {ROOT}/bin/*.sh", 60)
    link.write_text(f"code/{info['key']}.json", json.dumps({k: v for k, v in info.items() if k != "path"}, indent=1))
    if make_current:
        link.write_text("code/CURRENT", info["key"])
    info["upload"] = r
    return info


# ------------------------------------------------------------------------------------------------ models and weights


def model_items() -> list[tuple[Path, str]]:
    """What only this PC has and a pod cannot fetch (uploaded ONCE, hash-diffed): the gated SMPL-X / MAMMA locked-head body
    models, the downsampled vertices, the state_dict-only copy of the ma_2d checkpoint (540 MB; identical tensors, see
    work/runpod/weights/ma_2d/*.json) and the ALIKED / LightGlue weights. SMPLX_NEUTRAL.npz of smplx/ is a hard link of the
    locked-head copy (same sha256), so the volume links it instead of storing it twice. Public weights (SAM 2.1, YOLO) are
    fetched by the pod itself (`python -m dancecap.front_full fetch-weights`). Gated models never enter git."""
    bm = remote_jobs.env.mamma_data() / "body_models"
    items = [(bm / "smplx_locked_head" / "smplx" / f"SMPLX_{g}.npz", f"models/mamma_data/body_models/smplx_locked_head/smplx/SMPLX_{g}.npz")
             for g in ("NEUTRAL", "FEMALE", "MALE")]
    items += [(bm / "downsampled_verts" / "verts_512.pkl", "models/mamma_data/body_models/downsampled_verts/verts_512.pkl"),
              (bm / "downsampled_verts" / "smplx_512_body_parts.npz", "models/mamma_data/body_models/downsampled_verts/smplx_512_body_parts.npz"),
              (PACKAGE_ROOT / "work" / "runpod" / "weights" / "ma_2d" / "mamma_mask_full_cvpr.ckpt", "weights/mamma/ma_2d/mamma_mask_full_cvpr.ckpt")]
    hub = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
    for f in ("aliked-n16.pth", "aliked_lightglue_v0-1_arxiv.pth"):
        if (hub / f).exists():
            items.append((hub / f, f"weights/torch_home/hub/checkpoints/{f}"))
    missing = [str(p) for p, _ in items if not p.exists()]
    if missing:
        raise FileNotFoundError(f"model files missing locally: {missing}")
    return items


def sync_models(link: Link, parallel: int = 3, progress=None) -> dict:
    """Upload model_items() (resumable, skipped when the volume already holds the same sha256) over `parallel` ssh streams,
    in the order a first MAMMA fit needs them: SMPLX_NEUTRAL, the downsampled vertices, ALIKED/LightGlue, then the
    FEMALE/MALE locked-head models (ma_3d loads all three but dancecap fits are neutral) and last the 540 MB ma_2d
    checkpoint (front jobs only). Until the real FEMALE/MALE files have landed they are symlinks to NEUTRAL (listed in
    STUBS.txt, replaced atomically by the upload)."""
    from concurrent.futures import ThreadPoolExecutor
    items = model_items()

    def rank(it):
        rel = it[1]
        for i, key in enumerate(("SMPLX_NEUTRAL", "downsampled_verts", "torch_home", "SMPLX_FEMALE", "SMPLX_MALE", "ma_2d")):
            if key in rel:
                return i
        return 9
    items.sort(key=rank)
    t0 = time.time()
    lh = f"{ROOT}/models/mamma_data/body_models/smplx_locked_head/smplx"
    link.sh(f"mkdir -p {ROOT}/models/smplx {ROOT}/weights/dancecap {ROOT}/weights/mamma/ma_2d {ROOT}/weights/mamma/sam2 "
            f"{ROOT}/weights/mamma/yolo {ROOT}/weights/torch_home/hub/checkpoints {lh}", 60)
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = []
        for it in items:
            futs.append(ex.submit(link.push_file, it[0], it[1], progress=progress))
            if "SMPLX_NEUTRAL" in it[1]:  # stubs for the two models a neutral fit never reads, until their upload lands
                link.sh(f"cd {lh} && for g in FEMALE MALE; do [ -e SMPLX_$g.npz ] || ln -s SMPLX_NEUTRAL.npz SMPLX_$g.npz; done; "
                        f"echo 'SMPLX_FEMALE.npz and SMPLX_MALE.npz are STUBS (symlinks to NEUTRAL) while this file exists' > {lh}/STUBS.txt", 60)
        results = [f.result() for f in futs]
    link.sh(f"cd {ROOT}/models/smplx && for g in NEUTRAL FEMALE MALE; do ln -sfn ../mamma_data/body_models/smplx_locked_head/smplx/SMPLX_$g.npz SMPLX_$g.npz; done; "
            f"ln -sfn {ROOT}/weights/mamma {ROOT}/models/mamma_data/weights; "
            f"cd {lh} && [ -L SMPLX_FEMALE.npz ] || [ -L SMPLX_MALE.npz ] || rm -f STUBS.txt", 60)
    return {"files": len(results), "sent_bytes": sum(r["sent"] for r in results), "skipped": sum(r["skipped"] for r in results),
            "seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------------------------ worlds


def world_dirs(dance: str, world_id: str) -> tuple[Path, Path]:
    """(data tree, tracked manifest) of a world package on this PC (FARM.md section 3)."""
    return (PACKAGE_ROOT / "work" / "worlds" / dance / world_id, PACKAGE_ROOT / "worlds" / dance / world_id / "manifest.json")


def sync_world(link: Link, dance: str, world_id: str, progress=None) -> dict:
    """Put the world package on the volume at worlds/<world_id>/ (data tree + manifest.json). The manifest goes LAST, so a
    world whose manifest is on the volume is complete; a second call with the same manifest sha256 sends nothing."""
    data, tracked = world_dirs(dance, world_id)
    if not data.is_dir():
        raise FileNotFoundError(f"world data tree {data} not found (python -m dancecap.farm world build)")
    man = data / "manifest.json" if (data / "manifest.json").exists() else tracked
    rel_m = f"worlds/{world_id}/manifest.json"
    have = link.state([rel_m], {rel_m: man.stat().st_size}).get(rel_m)
    if have and have[1] == sha256_file(man):
        return {"world_id": world_id, "skipped": True}
    items = [(f, f"worlds/{world_id}/{f.relative_to(data).as_posix()}") for f in sorted(data.rglob("*"))
             if f.is_file() and f.relative_to(data).as_posix() != "manifest.json"]
    r = link.push_files(items, progress=progress)
    link.push_file(man, rel_m, force=True)
    return {"world_id": world_id, "skipped": False, **r}


# ------------------------------------------------------------------------------------------------ jobs


def submit_job(link: Link, spec: dict, inputs: dict[str, Path] | None = None, *, progress=None) -> dict:
    """Queue a job. Input files come from `inputs` ({workspace-relative destination path: local file}) and/or from
    spec['inputs'][i]['local'] (what `farm plan` writes); each goes to chunks/<id>/in/<path> (paths under worlds/ src/
    weights/ models/ go to the shared tree: content-addressed, uploaded once for all jobs), hash-diffed and striped. The spec
    that lands on the volume has no 'local' keys and a sha256 / bytes for every input (a planned sha256 that no longer
    matches the file is an error: the file changed after planning). queue/<id>.json is written LAST and atomically, so a worker
    never sees a job whose inputs are still uploading."""
    spec = json.loads(json.dumps(spec))
    jid = spec["id"]
    wanted: dict[str, tuple[Path, dict]] = {}
    for inp in spec.get("inputs", []):
        loc = inp.pop("local", None)
        if loc:
            wanted[inp["path"]] = (Path(loc), inp)
    for ws_path, loc in (inputs or {}).items():
        wanted.setdefault(ws_path, (Path(loc), {"path": ws_path}))
    rows = {i["path"]: i for i in spec.get("inputs", [])}
    items = []
    for ws_path, (loc, row) in wanted.items():
        if not loc.is_file():
            raise FileNotFoundError(f"{jid}: input {ws_path}: local file {loc} is missing")
        digest, size = sha256_file(loc), loc.stat().st_size
        if row.get("sha256") and row["sha256"] != digest:
            raise RuntimeError(f"{jid}: input {ws_path} changed since planning (sha256 {digest[:12]} != {row['sha256'][:12]})")
        shared = ws_path.startswith(SHARED_PREFIXES)
        items.append((loc, ws_path if shared else f"chunks/{jid}/in/{ws_path}"))
        rows[ws_path] = {**row, "path": ws_path, "sha256": digest, "bytes": size}
    sent = link.push_files(items, progress=progress) if items else {}
    spec["inputs"] = list(rows.values())
    link.write_text(f"queue/{jid}.json", json.dumps(spec, indent=1))
    return {"id": jid, "inputs": len(items), "push": sent}


def job_status(link: Link, jid: str) -> dict:
    """done.json / failed.json / heartbeat / attempts / step markers of one chunk job."""
    base = f"chunks/{jid}"
    return {"done": link.read_json(f"{base}/done.json"), "failed": link.read_json(f"{base}/failed.json"),
            "heartbeat": link.read_json(f"{base}/heartbeat"), "attempts": link.read_json(f"{base}/attempts.json"),
            "manifest": link.read_json(f"{base}/out/manifest.json")}


def pull_result(link: Link, jid: str, dest_root: Path, *, extract: bool = True) -> dict:
    """Download chunks/<id>/out/result.tar (resumable, sha256-checked against out/manifest.json), extract it under
    `dest_root` (the local workspace: files land at their workspace-relative paths) and verify every member's sha256."""
    man = link.read_json(f"chunks/{jid}/out/manifest.json")
    if not man:
        raise RuntimeError(f"{jid}: no out/manifest.json on the volume")
    dest_root = Path(dest_root)
    tar_local = dest_root / "_pull" / f"{jid}.tar"
    r = link.pull_file(f"chunks/{jid}/out/{man['tar']['name']}", tar_local, size=man["tar"]["bytes"], sha256=man["tar"]["sha256"])
    bad = []
    if extract:
        with tarfile.open(tar_local) as tar:
            for m in tar.getmembers():
                n = m.name[2:] if m.name.startswith("./") else m.name
                if not m.isfile() or n.startswith("/") or ".." in Path(n).parts:
                    continue
                target = dest_root / n
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + ".part")
                with open(tmp, "wb") as fh:
                    fh.write(tar.extractfile(m).read())
                os.replace(tmp, target)
        want = {f["path"]: f["sha256"] for f in man["files"]}
        for rel, h in want.items():
            f = dest_root / rel
            if not f.exists() or sha256_file(f, use_cache=False) != h:
                bad.append(rel)
        if bad:
            raise RuntimeError(f"{jid}: {len(bad)} pulled files do not match the manifest, first {bad[0]}")
    tar_local.unlink(missing_ok=True)
    return {"id": jid, "files": len(man["files"]), "bytes": man["tar"]["bytes"], "pull": r, "missing": man.get("missing", []),
            "status": man.get("status")}


# ------------------------------------------------------------------------------------------------ describe / create


def describe(link: Link | None = None) -> dict:
    """Volume facts: farm_volume.json + the Runpod record (+ usage when a link is given)."""
    from . import pods
    vol = pods.volume_info()
    live = remote._ctl("network-volume", "get", vol["id"], check=False)
    out = {"volume": vol, "runpod": live if isinstance(live, dict) else str(live)[:200],
           "price_per_month_usd": round(vol["size_gb"] * vol.get("price_usd_per_gb_month", 0.07), 2)}
    if link:
        out["usage"] = link.disk()
        cur = link.read_text("env/CURRENT")
        out["env_current"] = cur.strip() if cur else None
        code = link.read_text("code/CURRENT")
        out["code_current"] = code.strip() if code else None
    return out


def create(name: str = "hm-farm", size_gb: int = 150, dc: str = "EU-RO-1", tier: str = "STANDARD") -> dict:
    """Create the volume and write farm_volume.json. STANDARD storage is $0.07/GB-month (first 1 TB), billed while the volume
    exists, attached or not. The volume is pinned to its data center (pods using it can only run there)."""
    if pods_json := (PACKAGE_ROOT / "farm_volume.json"):
        if pods_json.exists():
            raise RuntimeError(f"{pods_json} exists: volume already created")
    made = remote._ctl("network-volume", "create", "--name", name, "--size", str(size_gb), "--data-center-id", dc)
    doc = {"id": made["id"], "name": name, "dc": dc, "size_gb": size_gb, "tier": tier, "price_usd_per_gb_month": 0.07,
           "mount": "/workspace", "root": ROOT, "created": time.strftime("%Y-%m-%d")}
    (PACKAGE_ROOT / "farm_volume.json").write_text(json.dumps(doc, indent=2))
    return doc


def init_layout(link: Link) -> None:
    link.sh("mkdir -p " + " ".join(f"{ROOT}/{d}" for d in LAYOUT), 120)
