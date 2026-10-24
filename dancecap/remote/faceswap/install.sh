#!/usr/bin/env bash
# Pod setup for the splat-only face swap of a recorded film (dancecap.film_faceswap pod; remote/faceswap/pod_job_template.json).
# It runs as the job's first command, at pod-ready next to the bundle upload (remote_jobs.early_commands), so it reads nothing
# but dancecap/remote/ and the network:
#   - a uv python 3.13 venv ($FF_HOME/venv): FaceFusion v4 needs Python >= 3.13 (core.pre_check, queue.ShutDown); the image's
#     python 3.11 / torch stack is not used ("base": false)
#   - FaceFusion 57c01ed's requirements.txt pins (FF_PINS below; run.sh compares them with the shipped code's file) and
#     onnxruntime-gpu for this pod's driver: a CUDA 13 driver (>= 580) -> 1.30.0, the local env's version, with the CUDA 13 pip
#     libraries; an older driver -> FaceFusion's own 'cuda@12' choice, 1.24.4, with the CUDA 12 ones (cuDNN 9, cuBLAS, cuFFT ...)
#   - $FF_HOME/ffpy: that python with LD_LIBRARY_PATH on those pip libraries (the loader reads it when the process starts)
#   - a CUDA check: a one-conv ONNX model run on the CUDAExecutionProvider (cuDNN and cuBLAS really load), else exit 3: the
#     swap must never fall back to the CPU silently
#   - the FaceFusion models the swap uses (FF_MODELS = "<release>/<name>=<crc32 of the local copy> ...", ~1.5 GB) fetched from
#     FaceFusion's own download hosts (GitHub releases, Hugging Face as the fallback) into $FF_HOME/models in parallel; each
#     .onnx is checked against its .hash (FaceFusion's crc32) and that against the local one; run.sh links the folder into the
#     code's .assets/models, and FaceFusion's pre_check fetches anything still missing itself. Never over the home uplink.
# The FaceFusion code is the job's shipped `git archive` of 57c01ed (run.sh extracts it). Idempotent ($FF_HOME/install.ok).
set -uo pipefail
FF_HOME=${FF_HOME:-/workspace/ff}
VENV=$FF_HOME/venv
PY=$VENV/bin/python
MODELS=$FF_HOME/models
FF_PINS=${FF_PINS:-"numpy==2.4.6 onnx==1.23.1 opencv-python-headless==5.0.0.93 psutil==7.2.2 python-multipart==0.0.28 scipy==1.18.1 starlette==1.6.0 uvicorn==0.41.0 websockets==16.0"}
export UV_CACHE_DIR=$FF_HOME/uv-cache UV_LINK_MODE=hardlink UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-60}
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore
mkdir -p "$FF_HOME" "$MODELS"
cd "$FF_HOME" || exit 2
t0=$(date +%s)
el() { echo "$(( $(date +%s) - t0 ))s"; }
log() { echo "[ff-install $(el)] $*"; }
if [ -f "$FF_HOME/install.ok" ] && [ -x "$FF_HOME/ffpy" ]; then log "already installed: $(cat "$FF_HOME/install.ok")"; exit 0; fi

cpus=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r q p < /sys/fs/cgroup/cpu.max && [ "$q" != max ]; then cpus=$(( (q + p - 1) / p ))
elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) && [ "$q" -gt 0 ]; then
  p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000); cpus=$(( (q + p - 1) / p )); fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
CUDA_DRV=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: *\([0-9][0-9]*\.[0-9]*\).*/\1/p' | head -1)
log "cpus: quota $cpus, nproc $(nproc); RAM $(awk '/MemTotal/ {printf "%.0f GB", $2 / 1048576}' /proc/meminfo); driver CUDA ${CUDA_DRV:-?}; disk $(df -h /workspace | tail -1 | awk '{print $4}') free"
major=${CUDA_DRV%%.*}
ORT_SET=${FF_ORT:-auto}
if [ "$ORT_SET" = auto ]; then if [ "${major:-0}" -ge 13 ] 2>/dev/null; then ORT_SET=cuda@13; else ORT_SET=cuda@12; fi; fi
case $ORT_SET in
  cuda@13) ORT="onnxruntime-gpu[cuda,cudnn]==1.30.0"; LIBS="nvidia-cublas~=13.0" ;;
  cuda@12) ORT="onnxruntime-gpu[cuda,cudnn]==1.24.4"; LIBS="nvidia-cublas-cu12~=12.0" ;;
  *) log "unknown FF_ORT $ORT_SET (cuda@13 | cuda@12 | auto)"; exit 2 ;;
esac
log "onnxruntime set $ORT_SET: $ORT"

# 1. the models, in the background (curl, 3 at a time); FaceFusion's .hash = crc32 of the .onnx as 8 hex digits
fetch_one() {  # release name want
  local rel=$1 name=$2 want=$3 ok=""
  for base in "https://github.com/facefusion/facefusion-assets/releases/download/$rel" "https://huggingface.co/facefusion/$rel/resolve/main"; do
    if [ -s "$MODELS/$name.hash" ] && [ -s "$MODELS/$name.onnx" ]; then ok=1; break; fi
    curl -fsSL --retry 4 --retry-delay 3 --connect-timeout 20 -o "$MODELS/$name.hash.part" "$base/$name.hash" &&
      curl -fsSL --retry 4 --retry-delay 3 --connect-timeout 20 -o "$MODELS/$name.onnx.part" "$base/$name.onnx" &&
      mv "$MODELS/$name.hash.part" "$MODELS/$name.hash" && mv "$MODELS/$name.onnx.part" "$MODELS/$name.onnx" && { ok=1; break; }
    rm -f "$MODELS/$name.hash.part" "$MODELS/$name.onnx.part"
  done
  [ -n "$ok" ] || { echo "[ff-install] model $name: download failed (FaceFusion's pre_check will try again)"; return 1; }
  local got crc
  got=$(cat "$MODELS/$name.hash")
  crc=$(python3 -c "import sys, zlib; h = 0
with open(sys.argv[1], 'rb') as f:
    for b in iter(lambda: f.read(1 << 24), b''): h = zlib.crc32(b, h)
print(format(h, '08x'))" "$MODELS/$name.onnx")
  if [ "$crc" != "$got" ]; then echo "[ff-install] model $name: crc32 $crc != its .hash $got: removed"; rm -f "$MODELS/$name.onnx"; return 1; fi
  if [ -n "$want" ] && [ "$want" != "$got" ]; then echo "[ff-install] model $name: hash $got differs from the local copy's $want"; return 1; fi
  echo "[ff-install] model $name ($rel) $(stat -c %s "$MODELS/$name.onnx") B, crc32 $got${want:+ = the local copy}"
}
(
  n=0; fails=0; pids=()
  for item in ${FF_MODELS:-}; do
    spec=${item%%=*}; want=${item#*=}; [ "$want" = "$item" ] && want=""
    fetch_one "${spec%%/*}" "${spec#*/}" "$want" & pids+=($!)
    n=$((n + 1))
    if [ $(( n % 3 )) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p" || fails=$((fails + 1)); done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p" || fails=$((fails + 1)); done
  echo "[ff-install] models: $n fetched, $fails failed, $(du -sh "$MODELS" | cut -f1) in $MODELS"
  echo "$fails" > "$FF_HOME/models.rc"
) > "$FF_HOME/models.log" 2>&1 &
MODELS_PID=$!

# 2. uv + the python 3.13 venv + the pins + onnxruntime-gpu (one uv call)
command -v uv > /dev/null || python -m pip install -q uv || exit 4
[ -x "$PY" ] || uv venv -q --python 3.13 "$VENV" || exit 5
# shellcheck disable=SC2086
uv pip install -q --python "$PY" $FF_PINS "$ORT" $LIBS || { log "uv install failed; pip fallback"; "$PY" -m ensurepip -q 2>/dev/null; "$PY" -m pip install -q $FF_PINS "$ORT" $LIBS || exit 6; }
echo "$FF_PINS" > "$FF_HOME/pins.txt"
log "venv $("$PY" -V 2>&1): $("$PY" -c 'import numpy, cv2, scipy, onnxruntime as o; print("numpy", numpy.__version__, "cv2", cv2.__version__, "scipy", scipy.__version__, "onnxruntime", o.__version__)')"

# 3. ffpy: the venv python with the pip CUDA libraries on LD_LIBRARY_PATH
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
LIBDIRS=$(find "$SP/nvidia" -name 'lib*.so*' -printf '%h\n' 2>/dev/null | sort -u | paste -sd: -)
cat > "$FF_HOME/ffpy" <<EOF
#!/usr/bin/env bash
# written by dancecap/remote/faceswap/install.sh ($ORT_SET)
export LD_LIBRARY_PATH="$LIBDIRS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=\${OMP_NUM_THREADS:-1} MALLOC_ARENA_MAX=\${MALLOC_ARENA_MAX:-4}
exec "$PY" "\$@"
EOF
chmod +x "$FF_HOME/ffpy"
log "ffpy: LD_LIBRARY_PATH += $(echo "$LIBDIRS" | tr ':' '\n' | sed "s#$SP/##" | paste -sd' ' -)"

# 4. the CUDA check (a conv: cuDNN; its im2col / gemm path: cuBLAS)
"$FF_HOME/ffpy" - <<'EOF' || { log "CUDA check FAILED: onnxruntime would run on the CPU"; exit 3; }
import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
w = numpy_helper.from_array(np.random.default_rng(0).random((16, 3, 3, 3), np.float32), "w")
g = helper.make_graph([helper.make_node("Conv", ["x", "w"], ["c"], pads=[1, 1, 1, 1]), helper.make_node("MatMul", ["c", "m"], ["y"])], "check",
                      [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 64, 64])],
                      [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 64, 64])],
                      [w, numpy_helper.from_array(np.eye(64, dtype=np.float32), "m")])
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
s = ort.InferenceSession(m.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
y = s.run(None, {"x": np.ones((1, 3, 64, 64), np.float32)})[0]
print(f"[ff-install] onnxruntime {ort.__version__} device {ort.get_device()}: available {ort.get_available_providers()}, "
      f"check session on {s.get_providers()}, output {y.shape} sum {float(y.sum()):.1f}", flush=True)
raise SystemExit(0 if s.get_providers()[0] == "CUDAExecutionProvider" else 3)
EOF

wait "$MODELS_PID"
cat "$FF_HOME/models.log"
echo "$ORT_SET $(date -u +%FT%TZ)" > "$FF_HOME/install.ok"
log "done"
