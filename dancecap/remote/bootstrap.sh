#!/usr/bin/env bash
# dancecap pod bootstrap (Linux Runpod pod, image runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04).
#
#   bash /workspace/HeadMovement/dancecap/remote/bootstrap.sh [feature ...]
#
# Features (base is always installed first):
#   base     apt: EGL/GL/GOMP for headless pyrender/open3d; torch 2.5.1+cu124 (matches local MAMMA); the
#            numeric + body-model stack pinned to the local .venv-nopo4d versions
#   mamma    MAMMA ma_3d requirements (minus Windows-only / unused GUI, SAM, detectron, rerun, wandb pieces) +
#            pytorch_sdf (cuevhv/pytorch_sdf@torch2.5-cu124, built on Linux for sm_80/sm_90)
#   room     gsplat 1.5.3 (no pt25 prebuilt wheel exists: built from the sdist for sm_80/sm_90), open3d, xatlas, plyfile
#   texture  nothing beyond base (avatar_raster is pure torch; no nvdiffrast)
#   dancers  SAM 2.1 (facebookresearch/sam2 at the commit installed locally, 2b90b9f) + hydra-core/iopath for the
#            dancers stage on a pod (dancecap.dancers_pod); its optional CUDA post-processing op is built
#            for sm_80/sm_90 when possible (SAM2_BUILD_ALLOW_ERRORS=1)
#   check    import checks (torch.cuda, smplx, pytorch_sdf, gsplat, open3d, pyrender/EGL) -> $STATE/check.json
#
# Idempotent: each feature leaves $STATE/<feature>.ok (with this script's version) and is skipped when present.
# Built CUDA wheels go to $WHEELHOUSE; when the runner uploads cached wheels there they are installed instead of
# rebuilt (dancecap/work/runpod/wheelhouse/<WHEEL_KEY>/ locally). Timings: $STATE/timings.tsv.
set -uo pipefail

BOOTSTRAP_VERSION=4
WS=${DANCECAP_WS:-/workspace/HeadMovement}
STATE=${DANCECAP_BOOTSTRAP_STATE:-/workspace/.dancecap_bootstrap}
WHEEL_KEY=cp311-torch2.5.1-cu124-sm80_90
WHEELHOUSE=${DANCECAP_WHEELHOUSE:-$WS/dancecap/work/runpod/wheelhouse/$WHEEL_KEY}
PY=${PY:-python}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_INPUT=1 DEBIAN_FRONTEND=noninteractive
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0"}
# nproc shows the HOST cores (256 on an A100 node): cap nvcc parallelism by the container's CPU quota and memory
_cpus=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r _q _p < /sys/fs/cgroup/cpu.max && [ "$_q" != max ]; then
  _cpus=$(( (_q + _p - 1) / _p ))
fi
_mem_gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1048576 ))
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
  _mem_gb=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
fi
_jobs=$(( _mem_gb / 4 )); [ $_jobs -gt $_cpus ] && _jobs=$_cpus; [ $_jobs -gt 32 ] && _jobs=32; [ $_jobs -lt 2 ] && _jobs=2
export MAX_JOBS=${MAX_JOBS:-$_jobs}
mkdir -p "$STATE" "$WHEELHOUSE"

log() { echo "[bootstrap] $*"; }
pipq() { "$PY" -m pip install -q "$@"; }
# pinned first (local versions); unpinned fallback so a missing wheel never blocks the session, but say so
pip_pinned() {
  if ! pipq "$@"; then
    log "WARNING: pinned install failed, retrying unpinned: $*"
    pipq $(printf '%s\n' "$@" | sed -E 's/[=<>]=?[^ ]*$//') || return 1
  fi
}

run_feature() {
  local f=$1 t0 rc
  if [ -f "$STATE/$f.ok" ] && grep -qx "v$BOOTSTRAP_VERSION" "$STATE/$f.ok"; then
    log "$f: already done"; return 0
  fi
  t0=$(date +%s)
  log "$f: start"
  "feature_$f"; rc=$?
  local dt=$(( $(date +%s) - t0 ))
  printf '%s\t%s\t%s\n' "$f" "$dt" "$rc" >> "$STATE/timings.tsv"
  if [ $rc -ne 0 ]; then log "$f: FAILED (rc=$rc) after ${dt}s"; return $rc; fi
  echo "v$BOOTSTRAP_VERSION" > "$STATE/$f.ok"
  log "$f: done in ${dt}s"
}

feature_base() {
  (apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 \
      libgomp1 libxrender1 pigz > /dev/null) || { log "apt failed"; return 1; }
  if ! "$PY" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1') and torch.version.cuda == '12.4' else 1)" 2>/dev/null; then
    pipq torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124 || return 1
  fi
  pip_pinned numpy==1.26.4 scipy==1.17.1 opencv-python-headless==4.11.0.86 PyYAML==6.0.3 tqdm pillow \
      matplotlib==3.11.1 smplx==0.1.28 trimesh==5.1.0 ninja jaxtyping rich || return 1
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('[bootstrap] torch', torch.__version__, torch.cuda.get_device_name(0))"
}

install_wheel_or_build() {  # $1 = dist name in wheel filename, $2 = pip spec to build from
  local name=$1 spec=$2 whl
  whl=$(ls "$WHEELHOUSE"/${name}-*.whl 2>/dev/null | head -1)
  if [ -n "$whl" ]; then
    log "$name: installing cached wheel $(basename "$whl")"
    pipq --no-deps --force-reinstall "$whl" && return 0
    log "$name: cached wheel failed to install; rebuilding"
  fi
  local t0=$(date +%s)
  log "$name: building wheel for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST with MAX_JOBS=$MAX_JOBS"
  "$PY" -m pip wheel -q --no-deps --no-build-isolation -w "$WHEELHOUSE" $spec || return 1
  log "$name: built in $(( $(date +%s) - t0 ))s"
  whl=$(ls -t "$WHEELHOUSE"/${name}-*.whl | head -1)
  pipq --no-deps --force-reinstall "$whl"
}

feature_mamma() {
  # requirements/requirements.txt minus GUI (flask, waitress), SAM2/SAM3, detectron2, ultralytics, open_clip, rerun,
  # wandb, TensorRT and VPoser: ma_3d with --skip_scene_videos imports only these
  pip_pinned pandas plotly==7.0.0 scenepic==1.1.2 pyrender==0.1.45 PyOpenGL==3.1.0 kornia==0.8.3 imageio-ffmpeg \
      scikit-learn loguru einops munkres yacs hydra-core psutil "setuptools<81" || return 1
  install_wheel_or_build pytorch_sdf "git+https://github.com/cuevhv/pytorch_sdf@torch2.5-cu124" || return 1
  "$PY" -c "from pytorch_sdf import sdf; print('[bootstrap] pytorch_sdf ok')"
}

feature_room() {
  # open3d -> dash -> flask -> blinker>=1.6, but Ubuntu ships a distutils blinker 1.4 pip cannot uninstall
  pipq --ignore-installed "blinker>=1.6" || return 1
  pip_pinned open3d==0.19.0 xatlas==0.0.11 plyfile==1.1 || return 1
  install_wheel_or_build gsplat "--no-binary gsplat gsplat==1.5.3" || return 1
  "$PY" -c "import gsplat; from gsplat.cuda._backend import _C; assert _C is not None; print('[bootstrap] gsplat', gsplat.__version__, 'CUDA ext ok')"
}

feature_texture() { return 0; }

feature_dancers() {
  pipq "hydra-core>=1.3.2" "iopath>=0.1.10" || return 1
  SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=1 pipq --no-deps --no-build-isolation       "git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4" || return 1
  "$PY" -c "import sam2; from sam2.build_sam import build_sam2_video_predictor; print('[bootstrap] sam2 ok', sam2.__file__)"
}

feature_check() {
  PYOPENGL_PLATFORM=egl "$PY" - "$STATE/check.json" <<'EOF'
import importlib, json, sys, time
out = {"t": time.time()}
def chk(name, fn):
    try:
        out[name] = {"ok": True, "info": fn()}
    except Exception as e:  # report every failure, don't stop at the first
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:400]}
def torch_cuda():
    import torch
    x = torch.randn(1024, 1024, device="cuda")
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)), "matmul": float((x @ x).sum().item() != 0),
            "mem_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)}
def sdf():
    import torch
    from pytorch_sdf import sdf as _s
    return {"module": _s.__name__ if hasattr(_s, "__name__") else str(type(_s))}
def gs():
    import gsplat, torch
    from gsplat.cuda._backend import _C
    from gsplat import rasterization
    n = 64
    means = torch.randn(n, 3, device="cuda") * 0.3 + torch.tensor([0, 0, 3.0], device="cuda")
    quats = torch.nn.functional.normalize(torch.randn(n, 4, device="cuda"), dim=-1)
    scales = torch.full((n, 3), 0.05, device="cuda"); op = torch.full((n,), 0.8, device="cuda")
    cols = torch.rand(n, 3, device="cuda")
    K = torch.tensor([[100.0, 0, 64], [0, 100.0, 64], [0, 0, 1]], device="cuda")[None]
    img, alpha, _ = rasterization(means, quats, scales, op, cols, torch.eye(4, device="cuda")[None], K, 128, 128)
    return {"gsplat": gsplat.__version__, "render_mean": float(img.mean())}
def smplx_():
    import smplx
    return {"smplx": getattr(smplx, "__version__", "?")}
def o3d():
    import open3d
    return {"open3d": open3d.__version__}
def pyr():
    import pyrender  # ma_3d imports it (never renders with --skip_scene_videos)
    return {"pyrender": pyrender.__version__}
def egl():
    import pyrender
    r = pyrender.OffscreenRenderer(64, 64); r.delete()
    return "egl offscreen ok"
for n, f in [("torch_cuda", torch_cuda), ("pytorch_sdf", sdf), ("gsplat", gs), ("smplx", smplx_),
             ("open3d", o3d), ("pyrender", pyr), ("egl_offscreen", egl)]:
    chk(n, f)
OPTIONAL = {"egl_offscreen"}  # informational: no stage renders with EGL on the pod today
import os
feats = set(os.environ.get("DANCECAP_FEATURES", "").split())
feats |= {f[:-3] for f in os.listdir(os.path.dirname(sys.argv[1])) if f.endswith(".ok")}  # installed earlier
if "mamma" not in feats:
    OPTIONAL |= {"pytorch_sdf", "pyrender", "pandas"}
if "room" not in feats:
    OPTIONAL |= {"gsplat", "open3d", "xatlas", "plyfile"}
out["features"] = sorted(feats)
for m in ("numpy", "scipy", "cv2", "trimesh", "xatlas", "plyfile", "pandas", "yaml"):
    chk(m, lambda m=m: getattr(importlib.import_module(m), "__version__", "ok"))
json.dump(out, open(sys.argv[1], "w"), indent=1)
bad = [k for k, v in out.items() if isinstance(v, dict) and not v.get("ok") and k not in OPTIONAL]
print("[bootstrap] check:", "ALL OK" if not bad else f"FAILED {bad}")
print(json.dumps(out, indent=1))
sys.exit(1 if bad else 0)
EOF
}

features=("base" "$@")
export DANCECAP_FEATURES="${features[*]}"
seen=" "
rc=0
for f in "${features[@]}"; do
  case "$seen" in *" $f "*) continue;; esac
  seen="$seen$f "
  if ! declare -F "feature_$f" > /dev/null; then log "unknown feature $f"; rc=2; continue; fi
  if [ "$f" = check ]; then rm -f "$STATE/check.ok"; fi  # checks always re-run
  run_feature "$f" || { rc=1; [ "$f" = base ] && break; }
done
log "finished rc=$rc (MAX_JOBS=$MAX_JOBS); wheels in $WHEELHOUSE: $(ls "$WHEELHOUSE" 2>/dev/null | tr '\n' ' ')"
exit $rc
