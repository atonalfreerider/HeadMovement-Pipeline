#!/usr/bin/env bash
# LHM++ (github.com/aigc3d/LHM-plusplus, Tongyi Lab; code Apache-2.0 + NVIDIA EG3D licence parts, weights CC BY-NC 4.0)
# at a pinned commit, in its OWN python 3.10 venv with the upstream-tested stack (INSTALL.md: torch 2.3.0 / torchvision
# 0.18.0 / CUDA 12.1 wheels, xformers 0.0.26.post1, spconv-cu121, torch_scatter 2.1.2+pt23cu121 (PyG), pytorch3d 0.7.6
# (Meta's py310_cu121_pyt230 wheel), gsplat 1.4.0+pt23cu121 (nerfstudio wheel), diff-gaussian-rasterization (ashawkey
# @ 8829d14, built here) and lib/pointops (built here)); the pod's system python (bootstrap 'base', torch 2.5.1) is
# left alone and runs the dancecap scoring step.
# Weights (downloaded on the pod in the background while the environment is built; they never leave the pod):
#   ModelScope Damo_XR_Lab/<LHMPP_MODEL> (model.safetensors, sha256 checked) and the needed part of HF 3DAIGC/LHMPP-Prior
#   (human_model_files without FLAME_texture / smplx_npz.zip / the Multi-HMR shape estimator, voxel_grid,
#   dense_sample_points, arcface; sha256 checked by aria2).
# Used by dancecap/remote/cand_lhmpp/pod_job.json (dancecap.mono4d.cand_lhmpp).
set -uo pipefail
COMMIT=${LHMPP_COMMIT:-906b5d9fb967ab42efb92f6fa55bf22cac86b653}
REPO=${LHMPP_ROOT:-/workspace/lhmpp_cand/LHM-plusplus}
VENV=${LHMPP_VENV:-/workspace/lhmpp_cand/venv}
MODEL=${LHMPP_MODEL:-LHMPP-700M-PixelShuffle}
BASE=$(dirname "$REPO")
DGR_COMMIT=8829d14f814fccdaf840b7b0f3021a616583c0a1
UV_VERSION=0.12.20
mkdir -p "$BASE"
cd "$BASE" || exit 2
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
export TORCH_CUDA_ARCH_LIST=${CAP:-8.6}
_cpus=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r _q _p < /sys/fs/cgroup/cpu.max && [ "$_q" != max ]; then _cpus=$(( (_q + _p - 1) / _p )); fi
export MAX_JOBS=$(( _cpus > 16 ? 16 : (_cpus < 2 ? 2 : _cpus) ))
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
echo "[install] cpus $_cpus MAX_JOBS $MAX_JOBS arch $TORCH_CUDA_ARCH_LIST nvcc $($CUDA_HOME/bin/nvcc --version | tail -2 | head -1)"

# ---------------------------------------------------------------- 1. weights in the background (aria2c, checksums)
(apt-get install -y -qq --no-install-recommends aria2 > /dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq aria2 > /dev/null 2>&1)) \
  && echo "[install] aria2 $(aria2c --version | head -1)"
PM=$REPO/pretrained_models
mkdir -p "$PM/Damo_XR_Lab/$MODEL"
LIST=$BASE/aria2_list.txt
MS=https://www.modelscope.cn/models/Damo_XR_Lab/$MODEL/resolve/master
HF=https://huggingface.co/3DAIGC/LHMPP-Prior/resolve/b683c8f68bede4f318b0bb539730b8e6711d30a0
{
  printf '%s\n  dir=%s\n  out=model.safetensors\n  checksum=sha-256=aa0750e7632352c50421c0e041f1e543277ce73a3af7ff26e446399394cac21e\n' "$MS/model.safetensors" "$PM/Damo_XR_Lab/$MODEL"
  printf '%s\n  dir=%s\n  out=config.json\n' "$MS/config.json" "$PM/Damo_XR_Lab/$MODEL"
  printf '%s\n  dir=%s\n  out=configuration.json\n' "$MS/configuration.json" "$PM/Damo_XR_Lab/$MODEL"
  while read -r f sha; do
    [ -z "$f" ] && continue
    printf '%s\n  dir=%s\n  out=%s\n' "$HF/$f" "$PM" "$f"
    [ "$sha" != "-" ] && printf '  checksum=sha-256=%s\n' "$sha"
  done <<'EOF'
arcface_resnet18.pth 69554b3184575bdda7ecf7f40caf2b0c5ecbd0617b676d2209d8bf0e7e3d8042
dense_sample_points/1_160000.ply b07679d85019cc493ed3b42b7711a4430e1d35754e5e93a64d13f2ec08550ec6
voxel_grid/cano_1_volume.npz c392f6a5c8c9ebde25365f2fb8cb8a66ae90e6330eb761ea7f59450cb6ac7bff
voxel_grid/human_prior_constrain.npz 52a9d4c2a107012a12491f07f1b5d0874c024d35cedef49def992c63927f41ea
voxel_grid/voxel_192.pth b94b46e1ebffe175117b3a1a2970d9551e07ebb08ff7c6649bdd20e8b75b46b6
human_model_files/flame/2019/female_model.pkl cf58da1da1bd72d01751339824a150ff739b5c3be809c8c1c992cccb44546e6b
human_model_files/flame/2019/generic_model.pkl e7f4f6ac7aa97045b3989893386aa94dc5245ded7105bbbf44116bfdf801a050
human_model_files/flame/2019/male_model.pkl 4ab6cb8f7a51c8eee4fb7ac9db10ccf3f61f0690f471b10919a5ae30addb8e47
human_model_files/flame/FLAME_FEMALEl.pkl efee38e24c9f9f7cc01120a5a48201a787e97e6e20af455747fd60f3dc06eb68
human_model_files/flame/FLAME_MALE.pkl 5a996eaca4b56b03a0c2907f7dfa3a92680d965fd190d345a7e588f6fa017d1b
human_model_files/flame/FLAME_NEUTRAL.pkl efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b
human_model_files/flame/flame_dynamic_embedding.npy fd046a4740f5e6837e65622d0e050273bb71bbcb1ad6ab322474013a2a23de37
human_model_files/flame/flame_static_embedding.pkl 2881bdad8e61e87aad83a3df2e04655c534694089f68ca7144e7350da9e8ad62
human_model_files/flame_assets/flame/FLAME_masks.pkl ccefbe1ac0774ff78c68caf2c627b4abc067a6555ebeb0be5d5b0812366ab492
human_model_files/flame_assets/flame/flame2023.pkl 8fb1af0db1abb51053ead8fd1f2624a63d01c9602f4a4fb4ea23bd2c82017fa0
human_model_files/flame_assets/flame/head_template_mesh.obj -
human_model_files/flame_assets/flame/landmark_embedding_with_eyes.npy a9050103cc98d5505beadca2f9b101acce9a78ebdd5d10cb001fa164ab7d309b
human_model_files/flame_assets/flame_arkit_bs.npy 33f4b588c1fadd227ce668c14b484619f40e7a08c25fdf20b0bc7b28212f43ca
human_model_files/flame_assets/pred_expression.json -
human_model_files/flame_assets/shoulder_mesh.obj -
human_model_files/flame_assets/teeth_jawopen_offset.npy da38cf36627942d8fea5a9157c4cb3788a2b7a0a9f331a40aac5f882e29cd616
human_model_files/mano/MANO_LEFT.pkl c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30
human_model_files/mano/MANO_RIGHT.pkl 45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767
human_model_files/smpl/SMPL_FEMALE.pkl 0e8c0bbbbc635dcb166ed29c303fb4bef16ea5f623e5a89263495a9e403575bd
human_model_files/smpl/SMPL_MALE.pkl a583c1b98e4afc19042641f1bae5cd8a1f712a6724886291a7627ec07acd408d
human_model_files/smpl/SMPL_NEUTRAL.pkl 98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42
human_model_files/smpl_mean_params.npz 6fd6dd687800da946d0a0492383f973b92ec20f166a0b829775882868c35fcdd
human_model_files/smplx/MANO_SMPLX_vertex_ids.pkl e5abe70b6574de25470475091e8008314a5b90127eb48c3e63bfa0adf8c04dcf
human_model_files/smplx/SMPL-X__FLAME_vertex_ids.npy 7e70cdc3659aae699b9732e8dd4af49106310c69b90dc83d9f73e96dbf871e49
human_model_files/smplx/SMPLX_FEMALE.npz b2a3686c9d6d218ff6822fba411c607a3c8125a70af340f384ce68bebecabe0e
human_model_files/smplx/SMPLX_MALE.npz ab318e3f37d2bfaae26abf4e6fab445c2a610e1d63714794d60379cc263bc2a5
human_model_files/smplx/SMPLX_NEUTRAL.npz 376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992
human_model_files/smplx/SMPLX_to_J14.pkl 5df844ddea85b0a400a2e8dbe63d09d19f2b1b7ec0e0e952daeae08f83d82d61
human_model_files/smplx/smplx_flip_correspondences.npz 0d3b9d42b3388457aa397d4580d207e0fc7e40da94b0d4b15f6a894fa808dedf
human_model_files/smplx/smplx_uv/smplx_uv.obj -
human_model_files/smplx/smplx_uv/smplx_uv.png f78cceddd979101a7e6981580d0838aed40b7807d7816741170e2edbc0994140
human_model_files/smplx/version.txt -
EOF
} > "$LIST"
DLLOG=$BASE/download.log
( if command -v aria2c > /dev/null; then
    aria2c -q -i "$LIST" -j 6 -x 16 -s 16 -k 8M --file-allocation=none -c --max-tries=8 --retry-wait=5 \
      --summary-interval=0 --console-log-level=warn > "$DLLOG" 2>&1
  else
    python3 - "$LIST" >> "$DLLOG" 2>&1 <<'PY'
import os, subprocess, sys
entries, cur = [], None
for line in open(sys.argv[1]):
    if not line.startswith(" "):
        cur = {"url": line.strip()}
        entries.append(cur)
    else:
        k, v = line.strip().split("=", 1)
        cur[k] = v
bad = 0
for e in entries:
    p = os.path.join(e["dir"], e["out"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    bad += subprocess.run(["curl", "-sSL", "--retry", "8", "-C", "-", "-o", p, e["url"]]).returncode != 0
sys.exit(1 if bad else 0)
PY
  fi
  echo "rc=$?" >> "$DLLOG"; echo "[install] downloads finished after $(el)" >> "$DLLOG" ) &
DL_PID=$!
( while kill -0 $DL_PID 2>/dev/null; do sleep 60; echo "[install] downloading: $(du -sm "$PM" 2>/dev/null | cut -f1) MB after $(el)"; done ) &
MON_PID=$!

# ---------------------------------------------------------------- 2. code at the pinned commit
if [ ! -d "$REPO/.git" ]; then
  rm -rf "$REPO.tmp" && git clone -q https://github.com/aigc3d/LHM-plusplus.git "$REPO.tmp" || exit 3
  # keep the pretrained_models dir the downloader is already writing into
  shopt -s dotglob; mv "$REPO.tmp"/* "$REPO"/ 2>/dev/null || cp -a "$REPO.tmp"/. "$REPO"/; shopt -u dotglob; rm -rf "$REPO.tmp"
fi
( cd "$REPO" && git checkout -q "$COMMIT" && echo "[install] LHM++ $(git log -1 --format='%H %cI')" ) || exit 4

# ---------------------------------------------------------------- 3. python 3.10 venv (uv-managed CPython, headers incl.)
python -m pip install -q "uv==$UV_VERSION" || exit 5
export UV_CACHE_DIR=$BASE/uv_cache UV_PYTHON_INSTALL_DIR=$BASE/uv_python UV_HTTP_TIMEOUT=120
[ -x "$VENV/bin/python" ] || uv venv -q --seed -p 3.10 --python-preference only-managed "$VENV" || exit 6
PY=$VENV/bin/python
UVP() { uv pip install -q --python "$PY" "$@"; }
UVP torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121 || exit 7
UVP --no-deps xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 || echo "[install] WARNING xformers failed (optional)"
cat > "$BASE/constraints.txt" <<EOF
torch==2.3.0
torchvision==0.18.0
numpy==1.23.0
EOF
# requirements.txt minus torch/torchvision/xformers/gsplat (installed separately), the Gradio app (gradio, spaces), rembg
# (we supply masks), chumpy (pkl body models only), modelscope (we download ourselves), dna, ffmpegio
grep -v -E '^\s*#|^\s*$' "$REPO/requirements.txt" | grep -v -i -E '^(torch|torchvision|xformers|gsplat|gradio|spaces|rembg|chumpy|modelscope|dna|ffmpegio)\b' \
  > "$BASE/req_filtered.txt"
UVP -c "$BASE/constraints.txt" -r "$BASE/req_filtered.txt" addict packaging ninja iopath fvcore || { echo "[install] filtered requirements failed"; exit 8; }
UVP -c "$BASE/constraints.txt" spconv-cu121==2.3.8 || exit 9
UVP -c "$BASE/constraints.txt" "torch_scatter==2.1.2+pt23cu121" -f https://data.pyg.org/whl/torch-2.3.0+cu121.html || exit 10
UVP -c "$BASE/constraints.txt" "https://github.com/nerfstudio-project/gsplat/releases/download/v1.4.0/gsplat-1.4.0%2Bpt23cu121-cp310-cp310-linux_x86_64.whl" || exit 11
"$PY" -m pip install -q --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt230/download.html || exit 12
# chumpy: the FLAME / MANO .pkl files of the prior unpickle chumpy objects; its setup.py imports pip, so no build
# isolation (the seeded venv has pip); numpy 1.23 keeps its `from numpy import bool, ...` working
"$PY" -c "import chumpy" 2>/dev/null || "$PY" -m pip install -q --no-build-isolation --no-deps chumpy==0.70 || exit 23
# flash-attn: the config's point backbone (PTv3 / sonata) has enable_flash: true; without the package it falls back to
# dense per-patch attention (N', 16, 4096, 4096) that ran a 24 GB card out of memory on 2026-10-19. Official
# Dao-AILab wheel for torch 2.3 / CUDA 12 / cp310 (cxx11abi FALSE = the PyPI torch build)
"$PY" -c "import flash_attn" 2>/dev/null || UVP --no-deps "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3%2Bcu123torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"   || echo "[install] WARNING flash-attn wheel failed: dense attention fallback (needs a 48 GB card)"
echo "[install] wheels done after $(el)"

# ---------------------------------------------------------------- 4. CUDA extensions built here (one arch)
if ! "$PY" -c "import diff_gaussian_rasterization" 2>/dev/null; then
  rm -rf "$BASE/dgr" && git clone -q --recursive https://github.com/ashawkey/diff-gaussian-rasterization "$BASE/dgr" || exit 13
  ( cd "$BASE/dgr" && git checkout -q "$DGR_COMMIT" && git submodule update -q --init --recursive ) || exit 14
  "$PY" -m pip install -q --no-build-isolation "$BASE/dgr" > "$BASE/build_dgr.log" 2>&1 || { tail -30 "$BASE/build_dgr.log"; exit 15; }
fi
if ! "$PY" -c "import torch, pointops_cuda" 2>/dev/null; then
  ( cd "$REPO/lib/pointops" && "$PY" -m pip install -q --no-build-isolation . > "$BASE/build_pointops.log" 2>&1 ) || { tail -30 "$BASE/build_pointops.log"; exit 16; }
fi
echo "[install] extensions built after $(el)"

# ---------------------------------------------------------------- 5. import check (+ pip names for missing modules)
cd "$REPO" || exit 17
for attempt in 1 2 3 4 5 6; do
  out=$("$PY" - 2>&1 <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
os.environ.update({"APP_ENABLED": "1", "APP_MODEL_NAME": os.environ.get("LHMPP_MODEL", "LHMPP-700M-PixelShuffle"),
                   "APP_TYPE": "infer.human_lrm_a4o", "NUMBA_THREADING_LAYER": "omp"})
import torch, numpy, pytorch3d, gsplat, spconv, torch_scatter, diff_gaussian_rasterization, pointops_cuda
from pytorch3d.ops import knn_points
from gsplat.rendering import rasterization
x = torch.randn(1, 64, 3, device="cuda"); knn_points(x, x, K=2)
from core.models import model_dict
from core.utils.hf_hub import wrap_model_hub
import importlib.util
spec = importlib.util.spec_from_file_location("_tg", "scripts/inference/to_gs_ply.py")
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)  # the module lhmpp_pod.py loads
try:
    import xformers; xf = xformers.__version__
except Exception as e:  # noqa
    xf = f"none ({e})"
try:
    import flash_attn; xf += " flash_attn " + flash_attn.__version__
except Exception as e:  # noqa
    xf += f" flash_attn none ({e})"
print("[install] import ok: torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "numpy",
      numpy.__version__, "pytorch3d", pytorch3d.__version__, "gsplat", gsplat.__version__, "xformers", xf)
PY
)
  rc=$?
  echo "$out" | tail -3
  [ $rc -eq 0 ] && break
  mod=$(echo "$out" | grep -oP "No module named '\K[^'.]+" | tail -1)
  [ -z "$mod" ] && { echo "$out" | tail -25; exit 18; }
  case "$mod" in cv2) pkg=opencv-python-headless;; PIL) pkg=Pillow;; yaml) pkg=PyYAML;; skimage) pkg=scikit-image;;
    sklearn) pkg=scikit-learn;; *) pkg=$mod;; esac
  echo "[install] missing module $mod -> uv pip install $pkg"
  UVP -c "$BASE/constraints.txt" "$pkg" || exit 19
done
[ $rc -eq 0 ] || exit 20
"$PY" -m pip freeze > "$BASE/pip_freeze.txt" 2>/dev/null

# ---------------------------------------------------------------- 6. weights
wait $DL_PID; kill $MON_PID 2>/dev/null
cat "$DLLOG"
grep -q "rc=0" "$DLLOG" || { echo "[install] downloads failed"; exit 21; }
sz=$(stat -c %s "$PM/Damo_XR_Lab/$MODEL/model.safetensors" 2>/dev/null || echo 0)
[ "$sz" = 5226465708 ] || { echo "[install] model.safetensors size $sz"; exit 22; }
du -sh "$PM"/* | sed 's/^/[install] /'
"$PY" - <<'PY' || exit 24
import pickle
import chumpy  # noqa: F401
for f in ("flame/FLAME_NEUTRAL.pkl", "mano/MANO_RIGHT.pkl"):
    with open("pretrained_models/human_model_files/" + f, "rb") as fh:
        d = pickle.load(fh, encoding="latin1")
    print("[install] unpickled", f, len(d), "keys")
PY
echo "[install] done in $(el)"
