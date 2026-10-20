#!/usr/bin/env bash
# NAVER Multi-HMR 2 (github.com/naver/multi-hmr2, NAVER non-commercial licence) at a pinned commit, on top of the pod's
# torch 2.5.1 (bootstrap 'base'), in the pod's system python with numpy / torch held by a constraints file.
#   - DINOv3 code (facebookresearch/dinov3, DINOv3 licence) pre-cloned at a pinned commit into the torch.hub cache
#     ($TORCH_HOME/hub/facebookresearch_dinov3_main): Multi-HMR 2's encoder calls torch.hub.load('facebookresearch/dinov3',
#     'dinov3_vitl16', pretrained=False); the fine-tuned backbone weights are inside the Multi-HMR 2 checkpoint.
#   - anny==0.6.0 (Apache-2.0 code) from PyPI; its non-commercial 'smplx' topology data (anny2smplx.pth, NAVER
#     non-commercial) is fetched by anny itself from download.europe.naverlabs.com on first use.
#   - checkpoint multihmr2.pt (1,613,132,154 bytes) from download.europe.naverlabs.com, downloaded in the background
#     while the Anny model caches are built (both take minutes).
# Used by dancecap/remote/cand_multihmr2/pod_job.json (dancecap.mono4d.cand_multihmr2). Weights never leave the pod.
set -uo pipefail
COMMIT=${MHMR2_COMMIT:-f066532d096034700e5ba8b7b1ca655729d3ea6e}
DINO_COMMIT=${DINOV3_COMMIT:-6876159a11b4df116f30f667f8c9888617df0751}
ROOT=${MHMR2_ROOT:-/workspace/mhmr2_cand}
CKPT=$ROOT/checkpoints/multihmr2.pt
CKPT_URL=https://download.europe.naverlabs.com/ComputerVision/MultiHMR/multihmr2.pt
CKPT_BYTES=1613132154
export TORCH_HOME=${TORCH_HOME:-$ROOT/torch_home}
export ANNY_CACHE_DIR=${ANNY_CACHE_DIR:-$ROOT/anny_cache}
mkdir -p "$ROOT/checkpoints" "$TORCH_HOME/hub" "$ANNY_CACHE_DIR"
cd "$ROOT" || exit 2
t0=$(date +%s)
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "[install] cpus $(nproc) python $(python -V 2>&1)"

# 1. checkpoint in the background (size-checked below)
if [ "$(stat -c %s "$CKPT" 2>/dev/null || echo 0)" != "$CKPT_BYTES" ]; then
  ( wget -q -O "$CKPT.part" "$CKPT_URL" && mv "$CKPT.part" "$CKPT"; echo "[install] checkpoint wget rc=$? after $(( $(date +%s) - t0 )) s" ) \
    > "$ROOT/ckpt_wget.log" 2>&1 &
  WGET_PID=$!
else
  WGET_PID=""
fi

# 2. code at pinned commits
if [ ! -d multi-hmr2/.git ]; then git clone -q https://github.com/naver/multi-hmr2.git multi-hmr2 || exit 3; fi
( cd multi-hmr2 && git checkout -q "$COMMIT" && echo "[install] multi-hmr2 $(git log -1 --format='%H %cI')" ) || exit 4
DINO_DIR=$TORCH_HOME/hub/facebookresearch_dinov3_main
if [ ! -d "$DINO_DIR/.git" ]; then git clone -q https://github.com/facebookresearch/dinov3.git "$DINO_DIR" || exit 5; fi
( cd "$DINO_DIR" && git checkout -q "$DINO_COMMIT" && echo "[install] dinov3 $(git log -1 --format='%H %cI')" ) || exit 6

# 3. python packages (numpy / torch kept at the bootstrap's versions when possible)
cat > "$ROOT/constraints.txt" <<EOF
numpy==1.26.4
torch==2.5.1
torchvision==0.20.1
EOF
PIPLOG=$ROOT/pip_full.log
pipc() { python -m pip install -q "$@" >> "$PIPLOG" 2>&1; }
DEPS="anny==0.6.0 roma einops scikit-learn warp-lang torchmetrics termcolor ftfy omegaconf regex submitit safetensors requests pandas"
if pipc -c "$ROOT/constraints.txt" $DEPS && pipc -c "$ROOT/constraints.txt" --no-deps -e ./multi-hmr2; then
  echo "[install] packages installed (numpy 1.26.4 / torch 2.5.1 kept)"
else
  tail -30 "$PIPLOG"
  echo "[install] constrained install failed; retrying with torch pinned only"
  printf 'torch==2.5.1\ntorchvision==0.20.1\n' > "$ROOT/constraints_torch.txt"
  pipc -c "$ROOT/constraints_torch.txt" $DEPS && pipc --no-deps -e ./multi-hmr2 || { tail -40 "$PIPLOG"; exit 7; }
fi
python - <<'EOF' || exit 8
import torch, numpy, anny, roma, warp, multihmr2, importlib.metadata as md
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "numpy", numpy.__version__,
      "anny", md.version("anny"), "roma", md.version("roma"), "warp", warp.__version__, "multihmr2", multihmr2.__file__)
EOF

# 4. Anny model caches (the four models the run builds) while the checkpoint downloads
python - <<'EOF' || exit 9
import time, warnings, torch, anny
warnings.filterwarnings("ignore")
for name, kw in (("decoder body_model", dict(local_changes=True, pose_parameterization="root_relative",
                                             remove_unattached_vertices=False)),
                 ("decoder lowres", dict(local_changes=True, pose_parameterization="root_relative",
                                         remove_unattached_vertices=False, topology="notoes_collapse5pc")),
                 ("session world", dict(local_changes=True, pose_parameterization="root_relative_world",
                                        remove_unattached_vertices=False)),
                 ("smplx topology", dict(local_changes=True, pose_parameterization="root_relative", topology="smplx"))):
    t0 = time.time()
    m = anny.create_fullbody_model(**kw)
    print(f"[install] anny {name}: {m.bone_count} bones, {m.template_vertices.shape[0]} verts, {time.time() - t0:.0f} s", flush=True)
EOF
ls "$ANNY_CACHE_DIR/noncommercial" > /dev/null 2>&1 && echo "[install] anny non-commercial data: $(ls "$ANNY_CACHE_DIR/noncommercial" | tr '\n' ' ')"

# 5. checkpoint
if [ -n "$WGET_PID" ]; then wait "$WGET_PID"; fi
cat "$ROOT/ckpt_wget.log" 2>/dev/null
sz=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
[ "$sz" = "$CKPT_BYTES" ] || { echo "[install] checkpoint size $sz != $CKPT_BYTES"; exit 10; }
echo "[install] checkpoint sha256 $(sha256sum "$CKPT" | cut -d' ' -f1) ($sz bytes)"
python -m pip freeze > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(( $(date +%s) - t0 )) s"
