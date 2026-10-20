#!/usr/bin/env bash
# Kimodo (github.com/nv-tlabs/kimodo, Apache-2.0) at a pinned commit, in its own venv on top of the pod's torch 2.5.1
# (bootstrap 'base'), + the open SOMA weights nvidia/Kimodo-SOMA-RP-v1.2 (NVIDIA Open Model License, not gated) at a
# pinned revision. No text encoder: the Llama-3 based LLM2Vec is never downloaded (the run passes a zero stub).
# Used by dancecap/remote/cand_kimodo/pod_job.json (dancecap.mono4d.cand_kimodo).
set -uo pipefail
COMMIT=${KIMODO_COMMIT:-e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954}
HF_REPO=${KIMODO_HF:-nvidia/Kimodo-SOMA-RP-v1.2}
HF_REV=${KIMODO_HF_REV:-0e3c89024d896064b6851201373f7834784a386b}
ROOT=/workspace/kimodo_cand
mkdir -p "$ROOT"
cd "$ROOT" || exit 2
t0=$(date +%s)
if [ ! -d kimodo/.git ]; then
  git clone -q https://github.com/nv-tlabs/kimodo.git kimodo || exit 3
fi
( cd kimodo && git checkout -q "$COMMIT" && echo "[install] kimodo $(git log -1 --format='%H %cI')" ) || exit 4
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
if [ ! -x venv/bin/python ]; then
  python -m venv --system-site-packages venv || { python -m pip install -q virtualenv && \
    python -m virtualenv -q --system-site-packages venv; } || exit 5
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -q --upgrade pip "setuptools<81" wheel || exit 6
# MotionCorrection (Kimodo's C++ post-processing) is built by setup.py with cmake. A pip-installed cmake is a python
# entry point that pip's isolated build env cannot import (session 20261019-135203: "cmake --version" exit 1), so use
# the system cmake binary; fall back to pip cmake + --no-build-isolation.
NOISO=""
if ! /usr/bin/cmake --version > /dev/null 2>&1; then
  (apt-get install -y -qq --no-install-recommends cmake > /dev/null 2>&1 || \
   { apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq --no-install-recommends cmake > /dev/null 2>&1; }) || true
fi
if ! /usr/bin/cmake --version > /dev/null 2>&1; then
  python -m pip install -q cmake && NOISO="--no-build-isolation"
fi
echo "[install] python $(python -V 2>&1) cmake $(cmake --version | head -1) ${NOISO}"
try_install() { python -m pip install -q $NOISO -e ./kimodo "$@" >> "$ROOT/pip_full.log" 2>&1; }
mc=1
if try_install "numpy==1.26.4"; then
  echo "[install] kimodo + MotionCorrection installed (numpy 1.26.4 kept)"
elif try_install; then
  echo "[install] kimodo + MotionCorrection installed (numpy unpinned)"
else
  tail -40 "$ROOT/pip_full.log"
  echo "[install] install with the MotionCorrection extension failed; retrying without it (post-processing off)"
  mc=0
  SKIP_MOTION_CORRECTION_IN_SETUP=1 python -m pip install -q -e ./kimodo >> "$ROOT/pip_full.log" 2>&1 || \
    { tail -40 "$ROOT/pip_full.log"; exit 7; }
fi
python - <<'EOF' || exit 8
import torch, numpy, transformers, kimodo
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "numpy", numpy.__version__, "transformers", transformers.__version__, "kimodo", kimodo.__file__)
EOF
python -c "import motion_correction._motion_correction as m; print('[install] motion_correction ok', m.__file__)" \
  || echo "[install] motion_correction NOT importable (mc=$mc): post-processing will be skipped"
python - <<EOF || exit 9
from huggingface_hub import snapshot_download
p = snapshot_download("$HF_REPO", revision="$HF_REV", local_dir="$ROOT/ckpt/$(basename $HF_REPO)")
import os
print("[install] weights", p, sorted(os.listdir(p)))
EOF
python -m pip freeze > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(( $(date +%s) - t0 )) s"
