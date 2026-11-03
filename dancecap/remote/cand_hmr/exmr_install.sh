#!/usr/bin/env bash
# Pod setup for the ExMR test (dancecap.hmr_test; job work/lucas_luiza/jobs/pod/lucas_luiza_exmr.json, command "install",
# started at pod-ready next to the upload: it reads nothing but dancecap/remote/ and the network - SMPL-X and MANO come
# from the bundle later, in exmr_pod.py place-models).
#   - a uv python 3.12 venv ($EXMR_ROOT/venv) with torch 2.8.0+cu126 / torchvision 0.23.0+cu126 (ExMR documents cu128,
#     which needs driver >= 570; with --no_render there is no CUDA extension, so cu126 as for SAM 3D Body), then ExMR's
#     requirements.txt with its constraints.txt, all in ONE uv call (the step-by-step install is the fallback)
#   - ExMR (twehrbein/ExMR, GVHMR-derived licence, owner OK 2026-10-27) at the pinned commit with its REQUIRED submodule
#     third_party/sam-3d-body (DPVO is optional: not fetched); HaMeR and Sapiens2 code with --no-deps at ExMR's pins;
#     pip install -e ExMR
#   - selective weights (background, next to the install): exmr.safetensors + the rtmw-x onnx (fast-mode fallback) at
#     $EXMR_HF_REV, sapiens2_1b_pose (Sapiens2 licence, owner OK), the GATED SAM 3D Body checkpoint (HF_TOKEN from the
#     job secrets, never printed), HaMeR's checkpoint files from the authors' Space. Then ExMR's own
#     tools/download_models.py --verify checks every sha256 against the version ExMR was tested with, stores the pinned
#     DINOv3 hub code and fetches whatever is still missing at its own pins; then that hub code is import-checked as
#     torch.hub will load it (review finding 5). No YOLO11 (a placeholder: our boxes), no DPVO, no nvdiffrast.
#   - no apt-get of its own (review finding 3): cv2 (opencv-python) needs libGL + glib, which the bootstrap's hardened apt
#     (feature base: deadsnakes source dropped, 20 s timeouts) installs next to this script; the import check waits up to
#     10 min for them, and only when they are still missing runs ONE bounded apt-get with the bootstrap's options.
#     libturbojpeg / libEGL are not needed with --no_render (only exmr/vis reads frames with TurboJPEG / renders).
set -uo pipefail
ROOT=${EXMR_ROOT:-/workspace/exmr}
EXMR_COMMIT=${EXMR_COMMIT:-91191a1013c2431f20a28af7b4433b04fdf1fef8}
EXMR_HF_REV=${EXMR_HF_REV:-a9a9cbc0aa7aee23ca99064c771b4dd0c99d842b}
S3DB_HF_REV=${S3DB_HF_REV:-11aaa346c7204874a1cbafe3d39a979080b2c55a}
SAPIENS2_COMMIT=${SAPIENS2_COMMIT:-7e5bae88456ac418ff0e58e74106c9fe192055d4}
SAPIENS2_HF_REV=${SAPIENS2_HF_REV:-f5fed8b97b99698d5eea1d14ff0855d0b4c3f000}
HAMER_COMMIT=${HAMER_COMMIT:-3a01849f4148352e9260b69bf28b65d1671a4905}
HAMER_SPACE_REV=${HAMER_SPACE_REV:-aabc78ee4fd1f1ccf1a55fdd4225bf84df0e680a}
REPO=$ROOT/ExMR
VENV=$ROOT/venv
PY=$VENV/bin/python
CK=${EXMR_CHECKPOINT_DIR:-$REPO/checkpoints}
export TORCH_HOME=${TORCH_HOME:-$ROOT/torch_home} HF_HOME=${HF_HOME:-$ROOT/hf_home} PYTHONNOUSERSITE=1
export UV_CACHE_DIR=$ROOT/uv-cache UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-60}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore HF_HUB_DISABLE_PROGRESS_BARS=1
mkdir -p "$ROOT" "$TORCH_HOME/hub" "$HF_HOME"
cd "$ROOT" || exit 2
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "[install] cpus $(nproc); disk $(df -h /workspace | tail -1 | awk '{print $4}') free"
if [ -z "${HF_TOKEN:-}" ]; then echo "[install] HF_TOKEN is not set (job secret missing)"; exit 3; fi
echo "[install] HF_TOKEN is set (value not shown)"

# 1. ExMR at the pin (GitHub serves a commit by hash at --depth 1; a full clone is the fallback)
clone() {  # url dir rev
  if [ ! -d "$2/.git" ]; then
    { mkdir -p "$2" && git -C "$2" init -q && git -C "$2" remote add origin "$1" &&
      git -C "$2" fetch -q --depth 1 origin "$3" && git -C "$2" checkout -q FETCH_HEAD; } ||
      { rm -rf "$2"; git clone -q "$1" "$2" && git -C "$2" checkout -q "$3"; } || return 1
  fi
  [ "$(git -C "$2" rev-parse HEAD)" = "$3" ] && echo "[install] $(basename "$2") $(git -C "$2" log -1 --format='%H %cI')"
}
clone https://github.com/twehrbein/ExMR.git "$REPO" "$EXMR_COMMIT" || { echo "[install] ExMR clone failed"; exit 4; }
( cd "$REPO" && { git submodule update --init --depth 1 third_party/sam-3d-body ||
                  git submodule update --init third_party/sam-3d-body; } &&
  echo "[install] sam-3d-body $(git -C third_party/sam-3d-body rev-parse HEAD) (ExMR's pinned submodule)" ) \
  > "$ROOT/submodule.log" 2>&1 &
SUB_PID=$!

# 2. venv + huggingface_hub at ExMR's pinned version (the big install below then leaves it alone)
command -v uv > /dev/null || python -m pip install -q uv || exit 5
[ -x "$PY" ] || uv venv -q --python 3.12 "$VENV" || exit 6
UVI() { uv pip install -q --python "$PY" "$@"; }
HFV=$(grep -iE '^huggingface[_-]hub==' "$REPO/requirements.txt" | head -1)
UVI -c "$REPO/constraints.txt" "${HFV:-huggingface_hub}" || exit 7
echo "[install] venv $("$PY" -V 2>&1), $HFV at $(el)"

# 3. weights in the background: our pins, each file once, moved into ExMR's checkpoint layout (exmr/assets.py)
mkdir -p "$CK"
( "$PY" - "$CK" "$EXMR_HF_REV" "$SAPIENS2_HF_REV" "$S3DB_HF_REV" "$HAMER_SPACE_REV" <<'EOF'
import shutil, sys, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import hf_hub_download
ck, exmr_rev, sap_rev, s3db_rev, hamer_rev = Path(sys.argv[1]), *sys.argv[2:6]
FILES = [  # (repo, type, revision, file in the repo, path under checkpoints/)
    ("twehrbein/ExMR", "model", exmr_rev, "exmr.safetensors", "exmr/exmr.safetensors"),
    ("twehrbein/ExMR", "model", exmr_rev, "rtmw-x_384x288_fp16.onnx", "rtmw/rtmw-x_384x288_fp16.onnx"),
    ("facebook/sapiens2-pose-1b", "model", sap_rev, "sapiens2_1b_pose.safetensors", "sapiens2/sapiens2_1b_pose.safetensors"),
    ("facebook/sam-3d-body-dinov3", "model", s3db_rev, "model.ckpt", "sam-3d-body-dinov3/model.ckpt"),
    ("facebook/sam-3d-body-dinov3", "model", s3db_rev, "model_config.yaml", "sam-3d-body-dinov3/model_config.yaml"),
    ("facebook/sam-3d-body-dinov3", "model", s3db_rev, "assets/mhr_model.pt", "sam-3d-body-dinov3/assets/mhr_model.pt"),
    ("geopavlakos/HaMeR", "space", hamer_rev, "_DATA/hamer_ckpts/checkpoints/hamer.ckpt", "hamer/hamer_ckpts/checkpoints/hamer.ckpt"),
    ("geopavlakos/HaMeR", "space", hamer_rev, "_DATA/hamer_ckpts/model_config.yaml", "hamer/hamer_ckpts/model_config.yaml"),
    ("geopavlakos/HaMeR", "space", hamer_rev, "_DATA/hamer_ckpts/dataset_config.yaml", "hamer/hamer_ckpts/dataset_config.yaml"),
    ("geopavlakos/HaMeR", "space", hamer_rev, "_DATA/data/mano_mean_params.npz", "hamer/data/mano_mean_params.npz"),
]
def get(f):
    repo, kind, rev, name, rel = f
    dst = ck / rel
    if dst.exists():
        return f"present {rel}"
    t = time.time()
    with tempfile.TemporaryDirectory(dir=ck) as tmp:
        p = hf_hub_download(repo, name, repo_type=kind, revision=rev, local_dir=tmp)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(p, dst)
    return f"{repo}@{rev[:7]} {name}: {dst.stat().st_size / 1e6:.0f} MB in {time.time() - t:.0f} s"
with ThreadPoolExecutor(4) as ex:
    for f, fut in [(f, ex.submit(get, f)) for f in FILES]:
        try:
            print("[install]", fut.result(), flush=True)
        except Exception as e:   # download_models.py --verify below retries at its own pins (the message has no token)
            print(f"[install] {f[0]} {f[3]}: {type(e).__name__}: {str(e).splitlines()[0][:200]}", flush=True)
EOF
) > "$ROOT/weights.log" 2>&1 &
W_PID=$!

# 4. torch (cu126) + ExMR's requirements in ONE uv call; the step-by-step install when it fails
TORCH=(torch==2.8.0+cu126 torchvision==0.23.0+cu126)
if UVI --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match \
     "${TORCH[@]}" -r "$REPO/requirements.txt" -c "$REPO/constraints.txt"; then
  echo "[install] one uv call: torch + requirements at $(el)"
else
  echo "[install] the one uv call failed at $(el); step by step"
  UVI torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126 || exit 8
  printf 'torch==2.8.0\ntorchvision==0.23.0\n' > "$ROOT/torch_pin.txt"
  UVI -r "$REPO/requirements.txt" -c "$REPO/constraints.txt" -c "$ROOT/torch_pin.txt" || exit 9
fi
UVI --no-deps "git+https://github.com/geopavlakos/hamer.git@$HAMER_COMMIT" \
    "git+https://github.com/facebookresearch/sapiens2.git@$SAPIENS2_COMMIT" || exit 10
wait "$SUB_PID"; rc=$?
cat "$ROOT/submodule.log"
[ $rc -eq 0 ] || { echo "[install] sam-3d-body submodule failed"; exit 11; }
UVI --no-deps -e "$REPO" || exit 12
echo "[install] packages at $(el)"

# 4b. libGL + glib for cv2 from the bootstrap's apt (feature base); wait up to 10 min. Still missing: one bounded apt-get
#     with the bootstrap's options (dead mirrors give up after 20 s, the deadsnakes source dropped, sources.list only).
have_libs() { ldconfig -p | grep -q 'libGL\.so\.1' && ldconfig -p | grep -q 'libglib-2\.0\.so\.0'; }
w0=$(date +%s)
until have_libs || [ $(( $(date +%s) - w0 )) -ge 600 ]; do sleep 10; done
if have_libs; then
  echo "[install] libGL + glib present (waited $(( $(date +%s) - w0 )) s for the bootstrap's apt)"
else
  echo "[install] libGL / glib still missing after $(( $(date +%s) - w0 )) s: one bounded apt-get (the bootstrap's options)"
  tail -5 "${DANCECAP_BOOTSTRAP_STATE:-/workspace/.dancecap_bootstrap}/apt.log" 2>/dev/null
  rm -f /etc/apt/sources.list.d/*deadsnakes* 2>/dev/null
  AO=(-o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20 -o DPkg::Lock::Timeout=120)
  grep -qE '^[[:space:]]*deb[[:space:]]' /etc/apt/sources.list 2>/dev/null &&
    AO+=(-o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0)
  { timeout 300 apt-get "${AO[@]}" update -qq &&
    DEBIAN_FRONTEND=noninteractive timeout 300 apt-get "${AO[@]}" install -y -qq --no-install-recommends \
      libgl1 libglib2.0-0; } > "$ROOT/apt.log" 2>&1 || { echo "[install] WARNING: apt failed"; tail -5 "$ROOT/apt.log"; }
fi

# 5. import checks: torch + CUDA, cv2, the three feature extractors' modules (no weights yet), TurboJPEG
( cd "$REPO" && "$PY" - <<'EOF'
import torch, numpy, cv2
from exmr.preprocessing.body_features import _import_sam_3d_body
_import_sam_3d_body()
import exmr.pipeline, exmr.preprocessing.hand_features, exmr.preprocessing.keypoints.sapiens2
import exmr.preprocessing.keypoints.rtmw, exmr.preprocessing.visual_odometry
from turbojpeg import TurboJPEG
try:
    TurboJPEG(); tj = "ok"
except Exception as e:  # only the renderer reads frames with it (not used: --no_render)
    tj = f"unavailable ({type(e).__name__})"
x = torch.randn(256, 256, device="cuda")
print("[install] torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0), "numpy",
      numpy.__version__, "cv2", cv2.__version__, "exmr", exmr.__version__, "turbojpeg", tj, "matmul",
      bool(torch.isfinite(x @ x).all()), flush=True)
EOF
) || { echo "[install] import check failed"; exit 13; }
echo "[install] imports at $(el)"

# 6. weights: wait, then ExMR's own sha256 check (+ the pinned DINOv3 hub code, + anything missing at its pins)
wait "$W_PID"
cat "$ROOT/weights.log"
( cd "$REPO" && "$PY" tools/download_models.py --checkpoint_dir "$CK" --only exmr rtmw sapiens2 sam3d hamer --verify ) \
  || { echo "[install] download_models.py --verify failed (gated SAM 3D Body access or a changed file?)"; exit 14; }
# the pinned DINOv3 hub code as torch.hub loads it: its hubconf pulls dinov3.hub.* / dinov3.eval.depth.models, so a
# missing package shows here, not in the smoke step (no bytecode: the tree stays as download_models.py hashed it)
( cd "$TORCH_HOME/hub/facebookresearch_dinov3_main" &&
  PYTHONDONTWRITEBYTECODE=1 "$PY" -c "import hubconf; print('[install] dinov3 hubconf import ok')" ) \
  || { echo "[install] the pinned DINOv3 hub code does not import (a missing package?)"; exit 15; }
mkdir -p "$CK/yolo"
[ -e "$CK/yolo/yolo11m.pt" ] || : > "$CK/yolo/yolo11m.pt"    # Assets.check placeholder: the tracker is never run
du -sh "$CK" 2>/dev/null
uv pip freeze --python "$PY" > "$ROOT/pip_freeze.txt" 2>/dev/null
echo "[install] done in $(el)"
