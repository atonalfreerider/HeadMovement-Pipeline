#!/usr/bin/env bash
# Pod setup for mono4d track "sam3d_body" (dancecap.mono4d.cand_sam3d_body; dancecap/remote/cand_sam3d_body/pod_job.json).
#   - a uv python 3.12 venv ($S3DB_ROOT/venv) with torch 2.8.0+cu126: pymomentum-cpu 0.1.114 (needed by MHR.from_files,
#     which the official MHR -> SMPL-X converter imports) ships only cp312/cp313 wheels built for torch 2.8. The pod's
#     system python 3.11 / torch 2.5.1 (bootstrap 'base') is not used by this job.
#   - SAM 3D Body code (facebookresearch/sam-3d-body, SAM License) at a pinned commit; no detector / SAM / MoGe.
#   - DINOv3 code (facebookresearch/dinov3, DINOv3 licence) at 5aa7e93 (the last commit before the 3DB release,
#     2025-11-19) pre-cloned into the torch.hub cache ($TORCH_HOME/hub/facebookresearch_dinov3_main): 3DB calls
#     torch.hub.load('facebookresearch/dinov3', <name>, source='github', pretrained=False); its weights are in model.ckpt.
#   - MHR (facebookresearch/MHR, Apache-2.0) at a pinned commit (the converter: tools/mhr_smpl_conversion) + its release
#     assets (v1.0.1 assets.zip: lod*.fbx, compact_v6_1.model, corrective blendshapes) for MHR.from_files.
#   - the GATED checkpoint facebook/sam-3d-body-dinov3 (model.ckpt, assets/mhr_model.pt, model_config.yaml) at a pinned
#     revision, downloaded in the background with HF_TOKEN from the environment (pushed by the runner from the owner's
#     key file; never printed). Weights never leave the pod.
set -uo pipefail
ROOT=${S3DB_ROOT:-/workspace/s3db_cand}
S3DB_COMMIT=${S3DB_COMMIT:-b5c765a0d89d789985e186d396315e7590887b94}
HF_REPO=${S3DB_HF_REPO:-facebook/sam-3d-body-dinov3}
HF_REV=${S3DB_HF_REV:-11aaa346c7204874a1cbafe3d39a979080b2c55a}
DINO_COMMIT=${DINOV3_COMMIT:-5aa7e93ab42ae3526291f6b4e3fcd22637e03326}
MHR_COMMIT=${MHR_COMMIT:-877cba5ee61a5f77d1457bf4de93a66695132bd9}
MHR_REL=${MHR_ASSETS_RELEASE:-v1.0.1}
VENV=$ROOT/venv
PY=$VENV/bin/python
CK=$ROOT/ckpt/sam-3d-body-dinov3
export TORCH_HOME=${TORCH_HOME:-$ROOT/torch_home} HF_HOME=${HF_HOME:-$ROOT/hf_home} UV_LINK_MODE=copy
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
mkdir -p "$ROOT/ckpt" "$TORCH_HOME/hub" "$HF_HOME"
cd "$ROOT" || exit 2
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "[install] cpus $(nproc) system python $(python -V 2>&1); disk $(df -h /workspace | tail -1 | awk '{print $4}') free"
if [ -z "${HF_TOKEN:-}" ]; then echo "[install] HF_TOKEN is not set (job secret missing)"; exit 3; fi
echo "[install] HF_TOKEN is set (value not shown)"

# 1. uv + python 3.12 venv
python -m pip install -q uv || exit 4
[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV" || exit 5
UVI() { uv pip install -q --python "$PY" "$@"; }
UVI "huggingface_hub>=0.30" || exit 6
echo "[install] venv $("$PY" -V 2>&1) at $(el)"

# 2. gated checkpoint, in the background
if [ "$(stat -c %s "$CK/model.ckpt" 2>/dev/null || echo 0)" != 2109129346 ]; then
  ( HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" - "$HF_REPO" "$HF_REV" "$CK" <<'EOF'
import sys, time
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
t = time.time()
p = snapshot_download(repo_id=repo, revision=rev, local_dir=dst,
                      allow_patterns=["model.ckpt", "model_config.yaml", "assets/mhr_model.pt", "LICENSE", "README.md"])
print(f"[install] {repo}@{rev[:7]} downloaded to {p} in {time.time() - t:.0f} s", flush=True)
EOF
  ) > "$ROOT/hf_download.log" 2>&1 &
  HF_PID=$!
else
  HF_PID=""
fi

# 3. code at pinned commits
clone() {  # url dir rev
  [ -d "$2/.git" ] || git clone -q "$1" "$2" || return 1
  ( cd "$2" && git checkout -q "$3" && echo "[install] $(basename "$1" .git) $(git log -1 --format='%H %cI')" )
}
clone https://github.com/facebookresearch/sam-3d-body.git "$ROOT/sam-3d-body" "$S3DB_COMMIT" || exit 7
clone https://github.com/facebookresearch/MHR.git "$ROOT/MHR" "$MHR_COMMIT" || exit 8
clone https://github.com/facebookresearch/dinov3.git "$TORCH_HOME/hub/facebookresearch_dinov3_main" "$DINO_COMMIT" || exit 9

# 4. MHR release assets for MHR.from_files (pure urllib/zipfile; background)
if [ ! -f "$ROOT/MHR/assets/lod1.fbx" ]; then
  ( cd "$ROOT/MHR" && PYTHONPATH="$ROOT/MHR" "$PY" -m mhr.download_assets --release "$MHR_REL" --dest "$ROOT/MHR" ) \
    > "$ROOT/mhr_assets.log" 2>&1 &
  MHR_PID=$!
else
  MHR_PID=""
fi

# 5. torch 2.8.0 (cu126) + the 3DB / converter / dinov3-hub dependencies (torch held by a constraints file)
UVI torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 || exit 10
printf 'torch==2.8.0\ntorchvision==0.23.0\n' > "$ROOT/constraints.txt"
UVI -c "$ROOT/constraints.txt" "numpy<2.3" scipy "opencv-python-headless<4.13" pillow tqdm pytorch-lightning yacs \
    omegaconf roma timm einops braceexpand pyrootutils "smplx==0.1.28" "trimesh>=4.8.3,<5" scikit-learn ftfy regex \
    submitit termcolor torchmetrics safetensors || exit 11
UVI -c "$ROOT/constraints.txt" "pymomentum-cpu==0.1.114.post0" \
  || echo "[install] WARNING: pymomentum-cpu did not install; the converter will use its stub (faces from the tool's own assets)"
echo "[install] packages at $(el)"
"$PY" - <<'EOF' || exit 12
import torch, numpy, cv2, roma, yacs, omegaconf, pytorch_lightning, smplx, trimesh, sklearn
x = torch.randn(256, 256, device="cuda")
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "numpy", numpy.__version__,
      "cv2", cv2.__version__, "lightning", pytorch_lightning.__version__, "smplx ok, matmul", bool(torch.isfinite(x @ x).all()))
EOF
"$PY" -c "import pymomentum.geometry, pymomentum.torch.character; print('[install] pymomentum import ok')" \
  || echo "[install] WARNING: pymomentum import failed (converter stub path)"
( cd "$TORCH_HOME/hub/facebookresearch_dinov3_main" && "$PY" -c "import hubconf; print('[install] dinov3 hubconf import ok')" ) || exit 13
( cd "$ROOT/sam-3d-body" && MOMENTUM_ENABLED=0 "$PY" -c "import sam_3d_body; print('[install] sam_3d_body import ok', sam_3d_body.__version__)" ) || exit 14

# 6. downloads
if [ -n "$HF_PID" ]; then wait "$HF_PID"; rc=$?; else rc=0; fi
tail -5 "$ROOT/hf_download.log" 2>/dev/null
for f in model.ckpt:2109129346 assets/mhr_model.pt:696110248; do
  sz=$(stat -c %s "$CK/${f%%:*}" 2>/dev/null || echo 0)
  if [ "$sz" != "${f##*:}" ]; then
    echo "[install] $CK/${f%%:*}: $sz bytes, expected ${f##*:} (hf download rc=$rc; gated access or token problem?)"; exit 15
  fi
done
[ -f "$CK/model_config.yaml" ] || { echo "[install] model_config.yaml missing"; exit 16; }
echo "[install] checkpoint ok: model.ckpt $(stat -c %s "$CK/model.ckpt") bytes, mhr_model.pt $(stat -c %s "$CK/assets/mhr_model.pt") bytes"
if [ -n "$MHR_PID" ]; then wait "$MHR_PID"; fi
tail -3 "$ROOT/mhr_assets.log" 2>/dev/null
ls "$ROOT/MHR/assets" 2>/dev/null | tr '\n' ' '; echo
[ -f "$ROOT/MHR/assets/lod1.fbx" ] || echo "[install] WARNING: MHR assets missing (converter stub path)"
"$PY" -m pip --version > /dev/null 2>&1 || true
uv pip freeze --python "$PY" > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(el)"
