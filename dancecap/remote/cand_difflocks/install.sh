#!/usr/bin/env bash
# DiffLocks (github.com/Meshcapade/difflocks, CVPR 2025; MPI non-commercial research licence for code + weights) at a
# pinned commit, in its own venv on top of the pod's torch 2.5.1/cu124 (bootstrap 'base'), for
# dancecap/remote/cand_difflocks/pod_job.json (dancecap.mono4d.cand_difflocks).
#   - requirements.txt minus torch/torchvision/opencv (the pod's torch 2.5.1 is kept: constraints file)
#   - flash-attn 2.7.4.post1 prebuilt wheel (cu12, torch 2.5, cxx11abi FALSE, cp311); on failure difflocks_pod.py
#     falls back to an SDPA implementation of the two functions DiffLocks calls
#   - NATTEN: not installed (PyPI has sdists only; difflocks_pod.py registers an exact pure-PyTorch na2d_qk/na2d_av)
#   - MediaPipe face landmarker on the CPU delegate (the GPU delegate needs a GLES context; one image per input)
#   - DINOv2 (torch.hub facebookresearch/dinov2) pre-cloned at a pinned commit + its ViT-L/14-reg weights
#   - the licensed checkpoints are linked from $DIFFLOCKS_CKPT_DIR (uploaded in the job bundle; the job's cleanup
#     command deletes them; the runner deletes the pod)
set -uo pipefail
COMMIT=${DIFFLOCKS_COMMIT:-fcc73747dc60320c30228b6711000a53fc0c9d84}
DINO_COMMIT=${DINOV2_COMMIT:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}
FA_WHL=${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}
CK=${DIFFLOCKS_CKPT_DIR:-/workspace/HeadMovement/models/difflocks/checkpoints}
ROOT=/workspace/difflocks_cand
export TORCH_HOME=${TORCH_HOME:-$ROOT/torch_home}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_INPUT=1 DEBIAN_FRONTEND=noninteractive
export PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-60} PIP_RETRIES=${PIP_RETRIES:-10}
mkdir -p "$ROOT" "$TORCH_HOME/hub"
cd "$ROOT" || exit 2
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
# MediaPipe imports sounddevice (PortAudio) and OpenCV contrib needs libGL
(apt-get install -y -qq --no-install-recommends libportaudio2 libgl1 libglib2.0-0 > /dev/null 2>&1 || \
  { apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq --no-install-recommends libportaudio2 libgl1 \
    libglib2.0-0 > /dev/null 2>&1; }) || echo "[install] apt libportaudio2 failed (continuing)"
if [ ! -d difflocks/.git ]; then
  git clone -q https://github.com/Meshcapade/difflocks.git difflocks || exit 3
fi
( cd difflocks && git checkout -q "$COMMIT" && echo "[install] difflocks $(git log -1 --format='%H %cI')" ) || exit 4
# MediaPipe GPU delegate -> CPU (no GLES context on the pod)
sed -i 's/delegate=mp.tasks.BaseOptions.Delegate.GPU/delegate=mp.tasks.BaseOptions.Delegate.CPU/' difflocks/inference/img2hair.py
grep -q "Delegate.CPU" difflocks/inference/img2hair.py || { echo "[install] delegate patch failed"; exit 5; }
# licensed checkpoints: links only (the files stay where the bundle put them)
for f in difflocks_diffusion/scalp_v9_40k_06730000_ema.pth strand_vae/strand_codec.pt strand_vae/hyperparams.json \
         rgb2material/rgb2material.pt rgb2material/hyperparams.json; do
  [ -f "$CK/$f" ] || { echo "[install] missing checkpoint $CK/$f"; exit 6; }
  mkdir -p "difflocks/checkpoints/$(dirname "$f")"
  ln -sf "$CK/$f" "difflocks/checkpoints/$f"
done
ls -la "$CK"/difflocks_diffusion/ | tail -n +2
if [ ! -x venv/bin/python ]; then
  python -m venv --system-site-packages venv || { python -m pip install -q virtualenv && \
    python -m virtualenv -q --system-site-packages venv; } || exit 7
fi
# shellcheck disable=SC1091
source venv/bin/activate
python -m pip install -q --upgrade pip wheel "setuptools<81" > /dev/null || exit 8
printf '%s\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 > "$ROOT/constraints.txt"
REQ="mediapipe==0.10.10 trimesh==4.0.4 jsonmerge==1.9.2 dctorch==0.1.2 einops==0.8.1 torchdiffeq==0.2.5 torchsde==0.2.6 libigl==2.5.1 safetensors accelerate==0.31.0"
if ! python -m pip install -q -c "$ROOT/constraints.txt" $REQ >> "$ROOT/pip_full.log" 2>&1; then
  tail -25 "$ROOT/pip_full.log"
  echo "[install] pinned requirements failed; retrying one by one (unpinned on failure)"
  for r in $REQ; do
    python -m pip install -q -c "$ROOT/constraints.txt" "$r" >> "$ROOT/pip_full.log" 2>&1 || \
      python -m pip install -q -c "$ROOT/constraints.txt" "${r%%==*}" >> "$ROOT/pip_full.log" 2>&1 || \
      { tail -20 "$ROOT/pip_full.log"; echo "[install] FAILED $r"; exit 9; }
  done
fi
if python -m pip install -q --no-deps "$FA_WHL" >> "$ROOT/pip_full.log" 2>&1 && \
   python -c "import flash_attn, flash_attn_2_cuda; print('[install] flash_attn', flash_attn.__version__)"; then
  :
else
  tail -8 "$ROOT/pip_full.log"
  python -m pip uninstall -y -q flash_attn > /dev/null 2>&1
  echo "[install] flash-attn wheel unusable: difflocks_pod.py uses the SDPA fallback"
fi
# DINOv2 for torch.hub at a pinned commit (torch.hub uses this cache dir instead of downloading main)
HUB=$TORCH_HOME/hub/facebookresearch_dinov2_main
if [ ! -d "$HUB/.git" ]; then
  rm -rf "$HUB"
  git clone -q https://github.com/facebookresearch/dinov2.git "$HUB" || exit 10
fi
( cd "$HUB" && git checkout -q "$DINO_COMMIT" && echo "[install] dinov2 $(git log -1 --format='%H %cI')" ) || exit 11
python - <<'EOF' || exit 12
import os, sys, time, torch
t = time.time()
m = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", trust_repo=True)
print("[install] dinov2_vitl14_reg ok", sum(p.numel() for p in m.parameters()) / 1e6, "M params", round(time.time() - t, 1), "s")
sys.path.insert(0, "/workspace/difflocks_cand/difflocks")
import mediapipe, igl, trimesh, numpy, cv2, dctorch, torchsde, torchdiffeq, jsonmerge
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "numpy",
      numpy.__version__, "mediapipe", mediapipe.__version__, "trimesh", trimesh.__version__, "cv2", cv2.__version__)
EOF
python -m pip freeze > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(( $(date +%s) - t0 )) s"
