"""Volume manifests (stdlib only, runs on the pod before/after the environment exists).

    python manifest.py env     --volume /workspace/hm [--env-dir /opt/hm/env]   # versions + sha256 of compiled wheels
    python manifest.py files   --volume /workspace/hm [--roots weights models] [--out weights/MANIFEST.sha256.json]

`env` writes <volume>/env/manifest.json: python, pip freeze, torch/cuda/gpu (when a GPU is visible), driver, image id, wheel
sha256s, the bootstrap state markers and timings. `files` writes a sha256 manifest of every file under the given roots
(a hash cache keyed by size+mtime keeps re-runs fast). Neither prints secrets (the environment is never dumped).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def sha256(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def _run(cmd: list[str], timeout: float = 120) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {e}"


def env_manifest(volume: Path, envd: Path | None = None) -> dict:
    envd = envd or Path("/opt/hm/env")
    py = envd / "bin" / "python"
    freeze = _run([str(py), "-m", "pip", "freeze", "--all"]).splitlines() if py.exists() else []
    torch = _run([str(py), "-c", "import json,torch;d={'torch':torch.__version__,'cuda':torch.version.cuda,"
                  "'gpu_visible':torch.cuda.is_available()};"
                  "d['gpu']=torch.cuda.get_device_name(0) if d['gpu_visible'] else None;print(json.dumps(d))"])
    try:
        torch = json.loads(torch)
    except json.JSONDecodeError:
        torch = {"error": torch[:200]}
    wheels = {}
    for w in sorted((volume / "wheels").rglob("*.whl")):
        wheels[str(w.relative_to(volume))] = {"bytes": w.stat().st_size, "sha256": sha256(w)}
    state = {}
    st = envd / ".state"
    if st.exists():
        for f in sorted(st.glob("*.ok")):
            state[f.stem] = f.read_text().strip()
        tt = st / "timings.tsv"
        if tt.exists():
            state["timings"] = [line.split("\t") for line in tt.read_text().splitlines()]
    out = {
        "schema": "dancecap.farm.env/1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "base_python": _run([str(py), "-c", "import sys;print(sys.base_prefix)"]) if py.exists() else None,
        "image": os.environ.get("RUNPOD_IMAGE_NAME") or os.environ.get("HM_IMAGE"),
        "torch": torch, "driver": _run(["nvidia-smi", "--query-gpu=driver_version,name,memory.total",
                                        "--format=csv,noheader"]).splitlines()[:1],
        "nvcc": (_run(["nvcc", "--version"]).splitlines() or [""])[-1],
        "arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST", "8.0;8.9;9.0"),
        "features": state, "wheels": wheels, "pip_freeze": freeze,
        "freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest(),
    }
    dest = volume / "env" / "manifest.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"[manifest] {dest}: {len(freeze)} packages, {len(wheels)} wheels, features {sorted(k for k in state if k != 'timings')}")
    return out


def files_manifest(volume: Path, roots: list[str], out: Path) -> dict:
    cache_p = volume / "tmp" / "sha_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    files = {}
    for r in roots:
        base = volume / r
        for f in sorted(base.rglob("*")):
            if f.is_file() and not f.is_symlink() and not f.name.endswith((".part", ".lock")):
                rel = str(f.relative_to(volume)).replace("\\", "/")
                st = f.stat()
                key = f"{st.st_size}:{int(st.st_mtime)}"
                c = cache.get(rel)
                digest = c["sha256"] if c and c["key"] == key else sha256(f)
                cache[rel] = {"key": key, "sha256": digest}
                files[rel] = {"bytes": st.st_size, "sha256": digest}
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache_p.write_text(json.dumps(cache))
    doc = {"schema": "dancecap.farm.files/1", "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "roots": roots, "n_files": len(files), "bytes": sum(v["bytes"] for v in files.values()), "files": files}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(f"[manifest] {out}: {len(files)} files, {doc['bytes'] / 1e9:.2f} GB")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["env", "files"])
    ap.add_argument("--volume", default="/workspace/hm")
    ap.add_argument("--env-dir", default="/opt/hm/env")
    ap.add_argument("--roots", nargs="*", default=["weights", "models"])
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    vol = Path(a.volume)
    if a.what == "env":
        env_manifest(vol, Path(a.env_dir))
    else:
        files_manifest(vol, a.roots, Path(a.out) if a.out else vol / "MANIFEST.files.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
