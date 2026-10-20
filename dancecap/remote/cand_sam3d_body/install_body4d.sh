#!/usr/bin/env bash
# Pod setup for the SAM-Body4D variants of mono4d track "sam3d_body" (job cand_sam3d_body4d, body4d_pod.py).
#   - runs install.sh first (idempotent: the venv, 3DB code + gated checkpoint, MHR converter are reused when the main
#     job already set them up in this pod)
#   - SAM-Body4D (gaomingqi/sam-body4d, MIT) at a pinned commit: its fork of SAM 3D Body (+ bundled dinov3 code),
#     Diffusion-VAS code, the app (scripts/offline_app.py). SAM 3 code is in the repo but never imported (bypassed).
#   - its extra Python deps (diffusers 0.29.1 as pinned upstream, transformers, accelerate, imageio, matplotlib)
#   - public checkpoints, downloaded on the pod at pinned revisions (background): Diffusion-VAS amodal segmentation +
#     content completion (kaihuac/*, MIT, 7.6 GB each), Depth-Anything-V2-Large (CC-BY-NC-4.0, 1.3 GB).
#     MoGe-2 / ViTDet / SAM 3 are NOT downloaded (our K, our masks).
set -uo pipefail
ROOT=${S3DB_ROOT:-/workspace/s3db_cand}
B4D_COMMIT=${B4D_COMMIT:-21af1020979ef32ddf6be3597ef59a68bad2f1bf}
VAS_MASK_REV=${VAS_MASK_REV:-7a79fa0}
VAS_RGB_REV=${VAS_RGB_REV:-428c072}
DA2_REV=${DA2_REV:-cbbb86a}
PY=$ROOT/venv/bin/python
CK=$ROOT/ckpt
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
HERE=$(cd "$(dirname "$0")" && pwd)

bash "$HERE/install.sh" || exit $?
echo "[install_b4d] base install ok at $(el)"
export TORCH_HOME=${TORCH_HOME:-$ROOT/torch_home} HF_HOME=${HF_HOME:-$ROOT/hf_home} UV_LINK_MODE=copy

# 1. public checkpoints in the background (resolve the short revisions to the full commit on the hub)
( HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" - "$CK" "$VAS_MASK_REV" "$VAS_RGB_REV" "$DA2_REV" <<'EOF'
import sys, time
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
ck, rm, rr, rd = sys.argv[1:5]
api = HfApi()
def full(repo, short):
    sha = api.model_info(repo).sha
    for c in api.list_repo_commits(repo):
        if c.commit_id.startswith(short):
            return c.commit_id
    print(f"[install_b4d] WARNING {repo}: revision {short} not found, using main {sha[:7]}", flush=True)
    return sha
t = time.time()
for repo, short, sub in (("kaihuac/diffusion-vas-amodal-segmentation", rm, "diffusion-vas-amodal-segmentation"),
                         ("kaihuac/diffusion-vas-content-completion", rr, "diffusion-vas-content-completion")):
    rev = full(repo, short)
    snapshot_download(repo_id=repo, revision=rev, local_dir=f"{ck}/{sub}")
    print(f"[install_b4d] {repo}@{rev[:7]} at {time.time() - t:.0f} s", flush=True)
rev = full("depth-anything/Depth-Anything-V2-Large", rd)
hf_hub_download("depth-anything/Depth-Anything-V2-Large", "depth_anything_v2_vitl.pth", revision=rev, local_dir=ck)
print(f"[install_b4d] Depth-Anything-V2-Large@{rev[:7]} at {time.time() - t:.0f} s", flush=True)
EOF
) > "$ROOT/b4d_download.log" 2>&1 &
DL_PID=$!

# 2. SAM-Body4D code at the pinned commit (shallow fetch of that commit)
B4D=$ROOT/sam-body4d
if [ ! -d "$B4D/.git" ]; then
  mkdir -p "$B4D" && ( cd "$B4D" && git init -q && git remote add origin https://github.com/gaomingqi/sam-body4d.git \
    && git fetch -q --depth 1 origin "$B4D_COMMIT" && git checkout -q FETCH_HEAD ) || exit 20
fi
( cd "$B4D" && echo "[install_b4d] sam-body4d $(git log -1 --format='%H %cI')" )

# 3. extra deps (torch held by the constraints file install.sh wrote)
uv pip install -q --python "$PY" -c "$ROOT/constraints.txt" "diffusers==0.29.1" "transformers~=4.57.3" \
    "accelerate~=1.12.0" "imageio[ffmpeg]" matplotlib pyyaml psutil || exit 21
echo "[install_b4d] packages at $(el)"
"$PY" "$HERE/body4d_pod.py" check --body4d-root "$B4D" || exit 22

# 4. downloads
wait "$DL_PID"; rc=$?
cat "$ROOT/b4d_download.log" | grep -v "^\s*$" | tail -6
[ "$rc" = 0 ] || { echo "[install_b4d] checkpoint download failed rc=$rc"; exit 23; }
for f in diffusion-vas-amodal-segmentation/unet/diffusion_pytorch_model.safetensors \
         diffusion-vas-content-completion/unet/diffusion_pytorch_model.safetensors depth_anything_v2_vitl.pth; do
  [ -s "$CK/$f" ] || { echo "[install_b4d] missing $CK/$f"; exit 24; }
done
du -sh "$CK"/* 2>/dev/null
uv pip freeze --python "$PY" > "$ROOT/pip_freeze_b4d.txt" 2>/dev/null
echo "[install_b4d] done in $(el)"
