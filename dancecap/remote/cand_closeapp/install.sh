#!/usr/bin/env bash
# CloseApp (github.com/boycehbz/CloseApp, CVPR 2025, MIT code) at a pinned commit on top of the pod's torch 2.5.1 / cu124
# (bootstrap 'base'), system python, numpy / torch held by a constraints file. Used by dancecap/remote/cand_closeapp/
# pod_job.json (dancecap.mono4d.cand_closeapp). Steps:
#   1. data.zip (README "Download data from ... Google Drive", 228,421,916 bytes: proxemics prior best_reconstruction.pkl,
#      UV / LBS maps, Halpe / H36M regressors, demo sequence 04305) in the background, size-checked
#   2. CloseApp code at CLOSEAPP_COMMIT
#   3. pip: lpips, pytorch-msssim, einops (pure python)
#   4. CUDA extensions for THIS GPU's arch only (TORCH_CUDA_ARCH_LIST from torch.cuda.get_device_capability):
#        diff-gaussian-rasterization (graphdeco-inria @ DGR_COMMIT, glm submodule as pinned; + #include <cstdint>)
#        SDF_ihmr (penincillin @ SDF_COMMIT; deprecated ATen calls patched: .data<T>() -> .data_ptr<T>(), .type())
#        torch-mesh-isect bvh_cuda (vchoutas @ ISECT_COMMIT; AT_CHECK -> TORCH_CHECK, .data<T>() -> .data_ptr<T>(),
#        helper_math.h from NVIDIA/cuda-samples v12.4 Common/ - the CUDA samples are not in the 12.x toolkit)
#      a failed build is reported, not fatal: closeapp_pod.py falls back (BVH: the penetration term has weight 0 in the
#      release, so a no-collision stand-in leaves the optimisation identical; SDF: only CloseApp's A-PD metric is lost;
#      rasterizer: the appearance variant cannot run)
#   5. data.zip unpacked into CloseApp/data, required files checked
set -uo pipefail
ROOT=${CLOSEAPP_ROOT:-/workspace/closeapp_cand}
COMMIT=${CLOSEAPP_COMMIT:-516ed6f8f8bd7bf5a358ce762e51845227d8aa85}
DGR_COMMIT=${DGR_COMMIT:-59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d}
SDF_COMMIT=${SDF_COMMIT:-8016aef2e8cd56f1b9d5481ddc0036543ad984c7}
ISECT_COMMIT=${ISECT_COMMIT:-85b30177821a1527e3fe62fcf8ce65262d7c1879}
DATA_ID=${CLOSEAPP_DATA_ID:-1kTMD6t-GSJ4tHnC2R19RD7dyMWMXwqsv}
DATA_BYTES=${CLOSEAPP_DATA_BYTES:-228421916}
HELPER_MATH_URL=https://raw.githubusercontent.com/NVIDIA/cuda-samples/v12.4/Common/helper_math.h
HELPER_MATH_SHA=b0e5e1e20960dbf64891d9c1578b4c69872d926063eba4081a6ce9df3daee124
mkdir -p "$ROOT"
cd "$ROOT" || exit 2
t0=$(date +%s)
STATUS=$ROOT/install_status.json
echo "{}" > "$STATUS"
setst() { python - "$STATUS" "$1" "$2" <<'EOF'
import json, sys
p, k, v = sys.argv[1:4]
d = json.load(open(p)); d[k] = v; json.dump(d, open(p, "w"), indent=1)
EOF
}
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
CAP=$(python -c "import torch; M, m = torch.cuda.get_device_capability(0); print(f'{M}.{m}')") || exit 3
export TORCH_CUDA_ARCH_LIST=$CAP
export FORCE_CUDA=1
echo "[install] cpus $(nproc) python $(python -V 2>&1) torch $(python -c 'import torch; print(torch.__version__, torch.version.cuda)') arch $CAP MAX_JOBS ${MAX_JOBS:-?}"
setst arch "$CAP"

# 1. data.zip (background)
ZIP=$ROOT/data.zip
if [ "$(stat -c %s "$ZIP" 2>/dev/null || echo 0)" != "$DATA_BYTES" ]; then
  ( for try in 1 2 3; do
      curl -sSL --retry 3 -o "$ZIP.part" "https://drive.usercontent.google.com/download?id=$DATA_ID&export=download&confirm=t"
      sz=$(stat -c %s "$ZIP.part" 2>/dev/null || echo 0)
      echo "[install] data.zip try $try: $sz bytes after $(( $(date +%s) - t0 )) s"
      if [ "$sz" = "$DATA_BYTES" ]; then mv "$ZIP.part" "$ZIP"; break; fi
      head -c 400 "$ZIP.part" 2>/dev/null | tr -cd '[:print:]\n' | head -5; sleep 5
    done ) > "$ROOT/data_dl.log" 2>&1 &
  DL_PID=$!
else
  DL_PID=""
fi

# 2. code
if [ ! -d CloseApp/.git ]; then git clone -q https://github.com/boycehbz/CloseApp.git CloseApp || exit 4; fi
( cd CloseApp && git checkout -q "$COMMIT" && echo "[install] CloseApp $(git log -1 --format='%H %cI')" ) || exit 5
setst closeapp "$COMMIT"

# 3. python packages
printf '%s\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 > "$ROOT/constraints.txt"
PIPLOG=$ROOT/pip.log
python -m pip install -q -c "$ROOT/constraints.txt" lpips==0.1.4 pytorch-msssim==1.0.0 einops==0.8.0 >> "$PIPLOG" 2>&1 \
  || { tail -20 "$PIPLOG"; exit 6; }
echo "[install] pip ok: $(python -c 'import lpips, pytorch_msssim, einops; print("lpips", lpips.__version__ if hasattr(lpips, "__version__") else "?", "einops", einops.__version__)')"

build() {  # name dir
  local name=$1 dir=$2 bt=$(date +%s)
  if (cd "$dir" && python -m pip install -q --no-build-isolation --no-deps . > "$ROOT/build_$name.log" 2>&1); then
    echo "[install] built $name in $(( $(date +%s) - bt )) s"; setst "build_$name" ok
  else
    echo "[install] BUILD FAILED $name after $(( $(date +%s) - bt )) s:"; grep -n "error" "$ROOT/build_$name.log" | head -20
    tail -15 "$ROOT/build_$name.log"; setst "build_$name" failed
  fi
}

# 4a. diff-gaussian-rasterization
if [ ! -d dgr/.git ]; then git clone -q https://github.com/graphdeco-inria/diff-gaussian-rasterization.git dgr; fi
( cd dgr && git checkout -q "$DGR_COMMIT" && git submodule update -q --init --recursive ) || echo "[install] dgr checkout failed"
if ! grep -q '#include <cstdint>' dgr/cuda_rasterizer/rasterizer_impl.h; then
  sed -i 's/#include <vector>/#include <vector>\n#include <cstdint>/' dgr/cuda_rasterizer/rasterizer_impl.h
fi
( cd dgr && echo "[install] dgr $(git log -1 --format=%H) glm $(git -C third_party/glm log -1 --format=%H)" )
build dgr dgr

# 4b. SDF_ihmr
if [ ! -d sdf_ihmr/.git ]; then git clone -q https://github.com/penincillin/SDF_ihmr.git sdf_ihmr; fi
( cd sdf_ihmr && git checkout -q "$SDF_COMMIT" \
  && sed -i 's/\.data<\([a-z_0-9]*\)>()/.data_ptr<\1>()/g; s/phi\.type()/phi.scalar_type()/g' sdf/csrc/sdf_cuda_kernel.cu \
  && sed -i 's/x\.type()\.is_cuda()/x.is_cuda()/g' sdf/csrc/sdf_cuda.cpp )
build sdf sdf_ihmr

# 4c. torch-mesh-isect (bvh_cuda)
if [ ! -d mesh_isect/.git ]; then git clone -q https://github.com/vchoutas/torch-mesh-isect.git mesh_isect; fi
mkdir -p "$ROOT/cuda_samples_inc"
if [ ! -f "$ROOT/cuda_samples_inc/helper_math.h" ]; then curl -sSL -o "$ROOT/cuda_samples_inc/helper_math.h" "$HELPER_MATH_URL"; fi
echo "$HELPER_MATH_SHA  $ROOT/cuda_samples_inc/helper_math.h" | sha256sum -c - || echo "[install] helper_math.h checksum differs"
export CUDA_SAMPLES_INC=$ROOT/cuda_samples_inc
# setup.py's '-Iinclude' is relative and ninja compiles from build/temp*: 2026-10-19 "double_vec_ops.h: No such file"
# -> the repo's own header goes next to helper_math.h (an absolute include dir). Added after that run (untested).
cp -f mesh_isect/include/double_vec_ops.h "$CUDA_SAMPLES_INC/" 2>/dev/null || true
( cd mesh_isect && git checkout -q "$ISECT_COMMIT" \
  && sed -i 's/AT_CHECK/TORCH_CHECK/g; s/x\.type()\.is_cuda()/x.is_cuda()/g' src/bvh.cpp \
  && sed -i 's/\.data<\([a-z_0-9]*\)>()/.data_ptr<\1>()/g; s/triangles\.type()/triangles.scalar_type()/g' src/bvh_cuda_op.cu )
build bvh mesh_isect

# 5. data
if [ -n "$DL_PID" ]; then wait "$DL_PID"; fi
cat "$ROOT/data_dl.log" 2>/dev/null
sz=$(stat -c %s "$ZIP" 2>/dev/null || echo 0)
[ "$sz" = "$DATA_BYTES" ] || { echo "[install] data.zip size $sz != $DATA_BYTES"; setst data failed; exit 10; }
echo "[install] data.zip sha256 $(sha256sum "$ZIP" | cut -d' ' -f1)"
if [ ! -f CloseApp/data/best_reconstruction.pkl ]; then
  python - "$ZIP" "$ROOT/CloseApp" <<'EOF' || exit 11
import sys, zipfile, os
zf = zipfile.ZipFile(sys.argv[1]); root = os.path.realpath(sys.argv[2])
for m in zf.infolist():  # members must stay inside CloseApp/
    dst = os.path.realpath(os.path.join(root, m.filename))
    if not dst.startswith(root + os.sep):
        raise SystemExit(f"refusing member {m.filename}")
zf.extractall(root)
print("[install] data.zip:", len(zf.infolist()), "members")
EOF
fi
for f in data/best_reconstruction.pkl data/smpl/J_regressor_halpe.npy data/lbs_map_smpl_512.npy data/smpl_faces.npy \
         data/uv_masks/uv_mask512_with_faceid_smpl.npy data/preprocess_data/04305/train/query_posemap_512_cano_smpl.npz \
         data/preprocess_data/04305/train/query_posemap_128_cano_smpl.npz data/preprocess_data/04305/train/smpl_cano_joint_mat.pth; do
  [ -f "CloseApp/$f" ] || { echo "[install] missing CloseApp/$f"; setst data failed; exit 12; }
done
setst data ok
python - <<'EOF'
import importlib, json
out = {}
for m in ("diff_gaussian_rasterization", "sdf", "bvh_cuda", "lpips", "pytorch_msssim", "einops"):
    try:
        importlib.import_module(m); out[m] = "ok"
    except Exception as e:  # noqa: BLE001
        out[m] = f"{type(e).__name__}: {e}"[:200]
print("[install] imports", json.dumps(out))
EOF
python -m pip freeze > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(( $(date +%s) - t0 )) s"
