#!/usr/bin/env bash
# Pod setup for the 4DAnyone test (dancecap.fda_test; job work/lucas_luiza/jobs/pod/lucas_luiza_fda.json, command "install").
#   - 4DAnyone (ant-research/4DAnyone, Apache-2.0 code) at the pinned commit with its GVHMR submodule at 4DAnyone's pin
#   - a uv python 3.11 venv ($FDA_ROOT/venv): 4DAnyone's requirements.txt (torch 2.8 / torchvision 0.23, cu128 wheels from
#     the default index), SDPA attention (no FlashAttention-3 / SageAttention build)
#   - the published weights at the pinned Hugging Face revision (AntResearch/4DAnyone @ HF_REVISION of fdanyone/assets.py,
#     ~21 GB: the 4DAnyone checkpoint (Apache-2.0), Wan2.2 VAE, the Turbo LoRA (CC BY-NC-SA 4.0), GVHMR (research licence),
#     HMR2, ViTPose, YOLOv8x (AGPL), BiRefNet) by 4DAnyone's own scripts/download_model.py, then
#   - fda_pod.py patch: the GVHMR tracker reads OUR dancer boxes when FDA_BOXES is set (one dancer per run).
# SMPL-X neutral (the owner's licence) is linked from the bundle by fda_pod.py place-models. Weights stay on the pod.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${FDA_ROOT:-/workspace/fda}
FDA_COMMIT=${FDA_COMMIT:-6dcf626d74bd95c42792f86b1d55e3841075c10c}
GVHMR_COMMIT=${GVHMR_COMMIT:-6ec3ca39336c50492c0fae65fba2fb831fc7d866}
REPO=$ROOT/4DAnyone
VENV=$ROOT/venv
export HF_HOME=${HF_HOME:-$ROOT/hf_home} UV_CACHE_DIR=$ROOT/uv-cache UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-90}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONNOUSERSITE=1
mkdir -p "$ROOT" "$HF_HOME"
cd "$ROOT" || exit 2
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "[install] cpus $(nproc); disk $(df -h /workspace | tail -1 | awk '{print $4}') free"

clone() {  # url dir rev
  if [ ! -d "$2/.git" ]; then
    { mkdir -p "$2" && git -C "$2" init -q && git -C "$2" remote add origin "$1" &&
      git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD; } ||
      { rm -rf "$2"; git clone -q "$1" "$2" && git -C "$2" checkout -q "$3"; } || return 1
  fi
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] && echo "[install] $(basename "$2") $(git -C "$2" log -1 --format='%H %cI')"
}
clone https://github.com/ant-research/4DAnyone.git "$REPO" "$FDA_COMMIT" || { echo "[install] 4DAnyone clone failed"; exit 4; }
rmdir "$REPO/third_party/GVHMR" 2>/dev/null || true
clone https://github.com/zju3dv/GVHMR.git "$REPO/third_party/GVHMR" "$GVHMR_COMMIT" || { echo "[install] GVHMR clone failed"; exit 4; }

if ! command -v uv >/dev/null 2>&1; then
  pip install -q uv || { echo "[install] uv install failed"; exit 5; }
fi
# uv's managed CPython (python-build-standalone ships tkinter): a GVHMR module has a stray `from turtle import forward`,
# and the image's system python 3.11 has no tkinter (session 20261105-123351: the motion worker died on it)
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 --python-preference only-managed "$VENV" || {
  echo "[install] venv failed"; exit 5; }
"$VENV/bin/python" -c "import turtle" || { echo "[install] tkinter / turtle missing in the venv python"; exit 5; }
echo "[install] venv $($VENV/bin/python -V) ($(el))"
( cd "$REPO" && uv pip install -q --python "$VENV/bin/python" -r requirements.txt ) || {
  echo "[install] requirements failed"; exit 6; }
uv pip install -q --python "$VENV/bin/python" lpips==0.1.4 || { echo "[install] lpips failed"; exit 6; }
echo "[install] requirements ($(el))"
"$VENV/bin/python" - <<'EOF' || { echo "[install] import check failed"; exit 7; }
import torch, torchvision, av, smplx, ultralytics
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
av.codec.Codec("libx264", "w"); av.codec.Codec("libx264rgb", "w")
print("[install] PyAV", av.__version__, "with libx264 / libx264rgb")
EOF
( cd "$REPO" && "$VENV/bin/python" scripts/download_model.py ) > "$ROOT/download_model.log" 2>&1 || {
  tail -20 "$ROOT/download_model.log"; echo "[install] model download failed"; exit 8; }
du -sh "$REPO/models" 2>/dev/null | sed 's/^/[install] models /'
echo "[install] weights ($(el))"
"$VENV/bin/python" "$HERE/fda_pod.py" patch || { echo "[install] tracker patch failed"; exit 9; }
"$VENV/bin/python" "$HERE/fda_pod.py" patch-render || { echo "[install] render patch failed"; exit 9; }
echo "[install] done ($(el))"
