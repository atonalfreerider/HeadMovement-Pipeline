"""Export and install the "sides" motion (DESIGN 4.1 `install.py`, 6.1 Phase D step 2).

    side_export(take, capture)      export_unity.export(..., install=False, smplx_dir="smplx_sides") into a SIDE capture
                                    (<capture>_sides, never installed), its four motion files copied to work/<take>/sides/unity_motion/ with
                                    unity_motion.json (sha256, fk_max_err_mm, the fit's sha256, the adopted windows), the proof against the
                                    input's export (prove_export) and install_motion.sh
    prove_export(udir, ref_dir)     outside the adopted windows + ramps and the prepare step's frames, every motion row byte-identical to the
                                    input's export; the follow bin changes only in spin / limit windows (and the splice)
    install(udir, capture_dir)      motion files only, under work/unity.lock, with a once-only backup of what is installed (--check: verify
                                    only; --restore: put the backup back). Never export(install=True) on a live capture: install_capture
                                    replaces the folder except splats.

The install is the orchestrator's (the Unity stage); this module only prepares and proves it.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .config import load_cfg
from .evidence import write_json

MOTION_FILES = ("smplx_lead.bin", "smplx_follow.bin", "figure1.json", "figure2.json")
BACKUP = "backup_installed_motion"


def log(*a) -> None:
    print("[sides.install]", *a, flush=True)


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def unity_motion_dir(take: str) -> Path:
    from ..take import PACKAGE_ROOT
    return PACKAGE_ROOT / "work" / take / "sides" / "unity_motion"


# ------------------------------------------------------------------------------------------------ the proof

def changed_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.flatnonzero((a != b).any(1))


def allowed_frames(gate: dict, prepare: dict, T: int) -> dict:
    """{role: (T,) bool} frames the sides fit may differ from the input on: adopted windows + ramps (their dancers), the splices (both)"""
    out = {"lead": np.zeros(T, bool), "follow": np.zeros(T, bool)}
    for w in gate.get("windows", {}).values():
        if w.get("status") != "adopt":
            continue
        for r in w.get("dancers", {}):
            out[r][w["r0"]:w["r1"] + 1] = True
    for s in prepare.get("splices", []):
        for r in out:
            out[r][s["k0"]:s["k1"]] = True
    for r, u in prepare.get("unwrap", {}).items():
        for ch in u.get("chains", {}).values():
            for run in ch.get("runs", []):
                out[r][max(0, run["a"] - 6):run["b"] + 7] = True
    return out


def prove_export(udir: Path, ref_dir: Path, gate: dict, prepare: dict, spin_frames: dict | None = None) -> dict:
    """the exported motion against the input's export, row by row: rows outside the allowed frames byte-identical; the follow bin
    changes only on spin / limit / splice frames; headers equal"""
    from .. import export_unity as eu
    out = {"ref_dir": str(ref_dir)}
    ok = True
    for r, f in (("lead", "smplx_lead.bin"), ("follow", "smplx_follow.bin")):
        h1, m1 = eu.read_motion(Path(udir) / f)
        h0, m0 = eu.read_motion(Path(ref_dir) / f)
        if m1.shape != m0.shape:
            out[r] = {"ok": False, "why": f"shape {m1.shape} vs {m0.shape}"}
            ok = False
            continue
        T = len(m0)
        allow = allowed_frames(gate, prepare, T)[r]
        ch = changed_rows(m1, m0)
        outside = ch[~allow[ch]]
        row = {"frames": int(T), "changed_rows": int(len(ch)), "changed_rows_outside_allowed": int(len(outside)),
               "outside_t": [round(x / 30.0, 2) for x in outside[:20]], "allowed_frames": int(allow.sum()),
               "header_equal": {k: (h1.get(k) == h0.get(k)) for k in sorted(set(h0) | set(h1))}}
        if r == "follow" and spin_frames is not None:
            nonspin = ch[~spin_frames[ch]]
            row["follow_changed_outside_spin_limit_splice"] = int(len(nonspin))
        row["ok"] = bool(len(outside) == 0 and all(row["header_equal"].values()) and row.get("follow_changed_outside_spin_limit_splice", 0) == 0)
        out[r] = row
        ok &= row["ok"]
    for f in ("figure1.json", "figure2.json"):
        same = (Path(udir) / f).read_bytes() == (Path(ref_dir) / f).read_bytes() if (Path(ref_dir) / f).exists() else None
        out[f] = {"identical_to_input_export": same}
    out["ok"] = bool(ok)
    return out


# ------------------------------------------------------------------------------------------------ the side export

def side_export(take: str, capture: str, smplx_dir: str = "smplx_sides", ref_dir: Path | None = None) -> dict:
    from .. import export_unity as eu
    cfg = load_cfg(take)
    work = Path(eu.PACKAGE_ROOT) / "work" / take
    side = f"{capture}_sides"
    t0 = time.time()
    rep = eu.export(take, side, install=False, room=False, textures=False, faces=False, shape_edits=True, allow_no_audio=True,
                    smplx_dir=smplx_dir)
    src = work / "export" / side
    udir = work / "sides" / "unity_motion"
    udir.mkdir(parents=True, exist_ok=True)
    for f in MOTION_FILES:
        shutil.copy2(src / f, udir / f)
    gate = json.loads((work / "sides" / "gate.json").read_text(encoding="utf-8"))
    prep = json.loads((work / "sides" / "cpu" / "prepare.json").read_text(encoding="utf-8"))
    ref = Path(ref_dir or (work / cfg.install_input_export_dir if cfg.install_input_export_dir else ""))
    T = None
    spin = None
    if (ref / "smplx_follow.bin").exists():
        hd, mm = eu.read_motion(ref / "smplx_follow.bin")
        T = len(mm)
        spin = np.zeros(T, bool)
        for w in gate.get("windows", {}).values():
            if w.get("status") == "adopt" and "follow" in w.get("dancers", {}) and ({"spin", "limit", "real"} & set(w.get("reasons", []))):
                spin[w["r0"]:w["r1"] + 1] = True
        for s in prep.get("splices", []):
            spin[s["k0"]:s["k1"]] = True
    out = {"what": "the sides motion (dancecap.sides): exported into the side capture, never installed by the export",
           "side_export": str(src), "files": [str(udir / f) for f in MOTION_FILES],
           "fk_max_err_mm": {r: d["fk_max_err_mm"] for r, d in rep["dancers"].items()}, "warnings": rep["warnings"],
           "sha256": {f: sha(udir / f) for f in MOTION_FILES},
           "fit": {r: {"path": str(work / smplx_dir / f"{r}.npz"), "sha256": sha(work / smplx_dir / f"{r}.npz")} for r in ("lead", "follow")},
           "adopted_windows": [{"id": int(i), "t0": w["t0"], "t1": w["t1"], "film0": w.get("film0"), "film1": w.get("film1"),
                                "dancers": sorted(w.get("dancers", {}))} for i, w in gate.get("windows", {}).items() if w.get("status") == "adopt"],
           "splices": prep.get("splices", []), "exported": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1)}
    if (ref / "smplx_lead.bin").exists():
        out["proof_vs_input_export"] = prove_export(udir, ref, gate, prep, spin)
    else:
        out["proof_vs_input_export"] = {"ok": None, "why": f"no input export at {ref} (cfg.install_input_export_dir)"}
    write_json(udir / "unity_motion.json", out)
    write_install_script(udir, capture)
    log("exported", out["sha256"], "proof ok:", out["proof_vs_input_export"].get("ok"))
    return out


# ------------------------------------------------------------------------------------------------ the install (the orchestrator runs it)

class LockHeld(RuntimeError):
    pass


def take_lock(lock: Path, owner: str) -> bool:
    """create work/unity.lock/ with owner.txt; False when someone else holds it"""
    try:
        lock.mkdir(parents=False)
    except FileExistsError:
        try:
            cur = (lock / "owner.txt").read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            cur = "?"
        if cur.strip() == owner.strip():
            return True
        return False
    (lock / "owner.txt").write_text(owner + "\n", encoding="utf-8")
    return True


def release_lock(lock: Path, owner: str) -> None:
    try:
        cur = (lock / "owner.txt").read_text(encoding="utf-8").splitlines()[0]
    except (OSError, IndexError):
        return
    if cur.strip() == owner.strip():
        shutil.rmtree(lock, ignore_errors=True)


def install(udir: Path, capture_dir: Path, check: bool = False, restore: bool = False, lock: Path | None = None,
            owner: str = "dancecap.sides install") -> dict:
    """motion files only (MOTION_FILES), sha256-checked against unity_motion.json; the first install backs up what is installed (never
    overwritten); restore = the backup back. Refuses while `lock` is held by someone else (the lock is taken for the copy unless
    check)."""
    udir, cap = Path(udir), Path(capture_dir)
    bak = udir / BACKUP
    if not cap.is_dir():
        raise FileNotFoundError(f"no capture folder {cap}")
    took = False
    if lock is not None and not check:
        if not take_lock(Path(lock), owner):
            raise LockHeld(f"{lock} is held by someone else")
        took = True
    try:
        if restore:
            if not (bak / "smplx_lead.bin").exists():
                raise FileNotFoundError(f"no backup in {bak}")
            for f in MOTION_FILES:
                shutil.copy2(bak / f, cap / f)
            return {"restored": True, "sha256": {f: sha(cap / f) for f in MOTION_FILES}}
        man = json.loads((udir / "unity_motion.json").read_text(encoding="utf-8"))
        bad = []
        rows = {}
        for f in MOTION_FILES:
            have = sha(udir / f)
            want = man["sha256"].get(f)
            if have != want:
                bad.append(f"MISMATCH {f}: {have} vs recorded {want}")
            inst = sha(cap / f) if (cap / f).exists() else None
            rows[f] = {"source": have, "installed": inst, "identical": inst == have}
        for f in ("smplx_lead.bin", "smplx_follow.bin"):
            if (cap / f).exists() and (udir / f).stat().st_size != (cap / f).stat().st_size:
                bad.append(f"SIZE {f}")
        if bad:
            raise RuntimeError("not installed: " + "; ".join(bad))
        if check:
            return {"check": "ok", "files": rows}
        bak.mkdir(parents=True, exist_ok=True)
        backed = False
        if not (bak / "smplx_lead.bin").exists():                 # never overwrite the first backup
            for f in (*MOTION_FILES, "capture.json"):
                if (cap / f).exists():
                    shutil.copy2(cap / f, bak / f)
            backed = True
        for f in MOTION_FILES:
            shutil.copy2(udir / f, cap / f)
        for f in MOTION_FILES:
            if sha(cap / f) != sha(udir / f):
                raise RuntimeError(f"copy of {f} failed")
        return {"installed": True, "backed_up_now": backed, "backup": str(bak), "sha256": {f: sha(cap / f) for f in MOTION_FILES}}
    finally:
        if took:
            release_lock(Path(lock), owner)


def write_install_script(udir: Path, capture: str) -> Path:
    """install_motion.sh next to the motion files (bash, LF): check / back up once / copy; --check; --restore; HM_CAPTURE_DIR overrides
    the capture (tests); refuses while work/unity.lock is held"""
    s = f'''#!/usr/bin/env bash
# For the UNITY STAGE (the orchestrator): put the sides motion (dancecap.sides) into the installed capture {capture}.
# Only the four motion files change; everything else in the capture (cameras/ phone clips, splats/, skins, audio, analyses, capture.json)
# stays as installed. The first install backs up the motion installed then (never overwritten), so --restore returns it.
#
#   bash <this folder>/install_motion.sh            # check, back up (first time only), then copy
#   bash <this folder>/install_motion.sh --check    # only check (sources, sizes, what is installed now)
#   bash <this folder>/install_motion.sh --restore  # put the backed-up motion back
#
# Each source file must have the sha256 recorded in unity_motion.json and each .bin the installed file's size. Run it while you hold the
# Unity editor (work/unity.lock); then reload the capture and re-record.
set -euo pipefail
SRC="$(cd "$(dirname "$0")" && pwd)"
WS="$(cd "$SRC/../../../../.." && pwd)"
CAP="${{HM_CAPTURE_DIR:-$WS/head-movement/Assets/StreamingAssets/{capture}}}"
LOCK="${{HM_UNITY_LOCK:-$WS/dancecap/work/unity.lock}}"
BAK="$SRC/{BACKUP}"
MAN="$SRC/unity_motion.json"
FILES="{' '.join(MOTION_FILES)}"
[ -d "$CAP" ] || {{ echo "no capture folder $CAP"; exit 1; }}
if [ -d "$LOCK" ] && [ "${{1:-}}" != "--check" ]; then
  owner="$(head -1 "$LOCK/owner.txt" 2>/dev/null || echo '?')"
  if [ "${{HM_UNITY_LOCK_OWNER:-}}" != "$owner" ]; then echo "work/unity.lock is held ($owner): not installed"; exit 1; fi
fi
if [ "${{1:-}}" = "--restore" ]; then
  [ -f "$BAK/smplx_lead.bin" ] || {{ echo "no backup in $BAK"; exit 1; }}
  for f in $FILES; do cp -p "$BAK/$f" "$CAP/$f"; done
  echo "restored the installed motion from $BAK"; exit 0
fi
[ -f "$MAN" ] || {{ echo "no $MAN (python -m dancecap.sides export --take <take>)"; exit 1; }}
bad=0
for f in $FILES; do
  want="$(grep -o "\\"$f\\": \\"[0-9a-f]\\{{64\\}}\\"" "$MAN" | head -1 | grep -o '[0-9a-f]\\{{64\\}}' || true)"
  have="$(sha256sum "$SRC/$f" | cut -d' ' -f1)"
  if [ -z "$want" ] || [ "$want" != "$have" ]; then echo "MISMATCH $f: sha256 $have, export recorded '${{want}}'"; bad=1; fi
  inst="$(sha256sum "$CAP/$f" 2>/dev/null | cut -d' ' -f1 || true)"
  if [ "$inst" = "$have" ]; then echo "  $f: the installed copy is identical"; else echo "  $f: the installed copy differs"; fi
done
for f in smplx_lead.bin smplx_follow.bin; do
  s1="$(stat -c %s "$SRC/$f")"; s0="$(stat -c %s "$CAP/$f")"
  if [ "$s1" != "$s0" ]; then echo "SIZE $f: $s1 bytes vs installed $s0 (frame count / layout differ)"; bad=1; fi
done
[ "$bad" = 0 ] || {{ echo "not installed: fix the export first"; exit 1; }}
[ "${{1:-}}" = "--check" ] && {{ echo "check ok"; exit 0; }}
mkdir -p "$BAK"
if [ ! -f "$BAK/smplx_lead.bin" ]; then                       # never overwrite the first backup: the motion installed before sides
  for f in $FILES capture.json; do [ -f "$CAP/$f" ] && cp -p "$CAP/$f" "$BAK/$f"; done
  echo "backed up the installed motion to $BAK"
fi
for f in $FILES; do cp "$SRC/$f" "$CAP/$f"; done
for f in $FILES; do
  [ "$(sha256sum "$CAP/$f" | cut -d' ' -f1)" = "$(sha256sum "$SRC/$f" | cut -d' ' -f1)" ] || {{ echo "copy of $f failed"; exit 1; }}
done
echo "installed the sides motion (dancecap.sides) into $CAP ($FILES)"
'''
    p = Path(udir) / "install_motion.sh"
    p.write_bytes(s.replace("\r\n", "\n").encode("utf-8"))
    return p
