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
#   gsplat   gsplat 1.5.3 (the cached wheel, else built as for room) + plyfile only: the dancer splats
#            (dancecap.dancer_splats) need no open3d / xatlas (a ~450 MB PyPI download avoided)
#   texture  nothing beyond base (avatar_raster is pure torch; no nvdiffrast)
#   dancers  SAM 2.1 (facebookresearch/sam2 at the commit installed locally, 2b90b9f) + hydra-core/iopath for the
#            dancers stage on a pod (dancecap.dancers_pod); its optional CUDA post-processing op is built
#            for sm_80/sm_90 when possible (SAM2_BUILD_ALLOW_ERRORS=1)
#   cameras  cameras stage on a pod (dancecap.front_pod): pycolmap 4.2.1 (CPU wheel, as locally) + LightGlue/ALIKED at the
#            local commit (eb42fee) + kornia
#   ma2d     MAMMA ma_2d inference (landmarks/run_ma_2d.py) on a pod: the lightning / albumentations / omegaconf stack it
#            imports, pinned to the local .venv-nopo4d versions (numpy held at 1.26.4)
#   ffmpeg   ffmpeg + ffprobe for jobs that decode video (the image has neither; 2026-10-21 trial: 2 sessions lost, one hung):
#            a static build (BtbN n8.1 = the local 8.1, else johnvansickle's release) fetched in the BACKGROUND while
#            base installs, Ubuntu's apt ffmpeg 4.4 as the fallback (dancecap.ingest handles both)
#   open3d   open3d 0.19.0 alone (physics_body.segment_masses): the blinker workaround of 'room' without gsplat / xatlas
#   check    import checks (torch.cuda, smplx, pytorch_sdf, gsplat, open3d, pyrender/EGL, ffmpeg) -> $STATE/check.json
#
# Installs go through uv (2026-10-22 probe on a 4090 pod, cold cache: base + mamma + open3d with torch 2.5.1+cu124 in 11 s,
# where pip took 278 s in the 2026-10-21 trial): ONE `uv pip install` of every requested feature's pins first (prefetch),
# then each feature's own steps, which find their packages in place; pip is the fallback of every install (DANCECAP_NO_UV=1
# forces pip). apt runs in the background next to the installs.
#
# Runner sessions start this script the moment the pod is up, before the bundle has arrived (remote_jobs parallel boot):
# DANCECAP_WAIT_BUNDLE=<marker> makes the steps that need bundle files (the cached CUDA wheels) wait for that marker.
# DANCECAP_CPU_POD=1 (a CPU pod): torch 2.5.1+cpu, no GPU check, no CUDA extensions (pytorch_sdf skipped; room, gsplat and
# dancers refuse).
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
# slow PyPI links on some community hosts (2026-10-07, one host: 15 s read timeouts x 5 retries failed 'mamma'):
# wait longer and retry more before a feature fails
export PIP_TIMEOUT=${PIP_TIMEOUT:-60} PIP_DEFAULT_TIMEOUT=${PIP_DEFAULT_TIMEOUT:-60} PIP_RETRIES=${PIP_RETRIES:-10}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;9.0"}
# nproc shows the HOST cores (256 on an A100 node): cap nvcc parallelism by the container's CPU quota and memory
_cpus=$(nproc)
if [ -r /sys/fs/cgroup/cpu.max ] && read -r _q _p < /sys/fs/cgroup/cpu.max && [ "$_q" != max ]; then
  _cpus=$(( (_q + _p - 1) / _p ))
fi
_mem_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo 2>/dev/null)   # absent on old kernels / Git Bash (the tests)
_mem_gb=$(( ${_mem_kb:-8388608} / 1048576 ))
if [ -r /sys/fs/cgroup/memory.max ] && [ "$(cat /sys/fs/cgroup/memory.max)" != max ]; then
  _mem_gb=$(( $(cat /sys/fs/cgroup/memory.max) / 1073741824 ))
fi
_jobs=$(( _mem_gb / 4 )); [ $_jobs -gt $_cpus ] && _jobs=$_cpus; [ $_jobs -gt 32 ] && _jobs=32; [ $_jobs -lt 2 ] && _jobs=2
export MAX_JOBS=${MAX_JOBS:-$_jobs}
mkdir -p "$STATE" "$WHEELHOUSE"

log() { echo "[bootstrap] $*"; }
cpu_pod() { [ "${DANCECAP_CPU_POD:-0}" = 1 ]; }
# the torch triple of this pod: CUDA 12.4 wheels (the local MAMMA's), or the CPU wheels on a CPU pod (175 MB, not 3 GB)
if cpu_pod; then TORCH_IDX=https://download.pytorch.org/whl/cpu; TV=+cpu; else TORCH_IDX=https://download.pytorch.org/whl/cu124; TV=+cu124; fi
TORCH_PINS=("torch==2.5.1$TV" "torchvision==0.20.1$TV" "torchaudio==2.5.1$TV")
torch_ok() {
  "$PY" -c "import torch, sys; sys.exit(0 if torch.__version__.startswith('2.5.1') and (torch.version.cuda == '12.4' or '$TV' == '+cpu') else 1)" 2>/dev/null
}
# uv next to pip: its cache on the target's filesystem (hardlinks, no second copy of 5 GB of wheels)
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.uv-cache} UV_LINK_MODE=${UV_LINK_MODE:-hardlink} UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-60}
UV=""
UVT=()
uv_ready() {
  [ -n "$UV" ] && return 0
  [ "${DANCECAP_NO_UV:-0}" = 1 ] && return 1
  command -v uv > /dev/null || "$PY" -m pip install -q uv > /dev/null 2>&1 || return 1
  UV=$(command -v uv) || return 1
  # a venv python (the farm env, hm_env.sh) is its own target; the image's python is the system environment
  if "$PY" -c 'import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)'; then UVT=(--python "$PY")
  else UVT=(--system --break-system-packages --python "$PY"); fi
}
uvq() {  # uv pip install with pip's flags translated (uv has no --ignore-installed: it never sees the distutils blinker)
  local a=() x
  for x in "$@"; do case "$x" in --ignore-installed) ;; --force-reinstall) a+=(--reinstall);; *) a+=("$x");; esac; done
  "$UV" pip install -q "${UVT[@]}" "${a[@]}"
}
# the parallel boot: steps that need a bundle file wait for the runner's marker (never longer than DANCECAP_WAIT_BUNDLE_S)
wait_bundle() {
  local m=${DANCECAP_WAIT_BUNDLE:-} i=0 lim=${DANCECAP_WAIT_BUNDLE_S:-1800}
  [ -n "$m" ] || return 0
  [ -e "$m" ] && return 0
  log "waiting for the bundle ($m)"
  while [ ! -e "$m" ] && [ $i -lt "$lim" ]; do sleep 1; i=$(( i + 1 )); done
  if [ -e "$m" ]; then log "bundle there after ${i}s"; else log "no bundle after ${i}s; going on without it"; fi
}
# bundled pure-python / manylinux wheels (dancecap/work/runpod/pywheels/cp311-linux, see remote_jobs.PYWHEELS) are
# tried first with --no-index: 2026-10-07 one host stalled on every large PyPI page (20 min bootstrap)
PYWHEELS=${DANCECAP_PYWHEELS:-$WS/dancecap/work/runpod/pywheels/cp311-linux}
pipq() {
  if [ -d "$PYWHEELS" ]; then
    if uv_ready && uvq --no-index --find-links "$PYWHEELS" "$@" 2>/dev/null; then return 0; fi
    "$PY" -m pip install -q --no-index --find-links "$PYWHEELS" "$@" 2>/dev/null && return 0
  fi
  if uv_ready; then
    uvq "$@" && return 0
    log "uv failed ($*); pip"
  fi
  "$PY" -m pip install -q "$@"
}
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

APT_PKGS="libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 libxrender1 pigz"
# 2026-10-21: the image's deadsnakes PPA answered 503 and `apt-get update` stalled 860 s on one pod (minutes on others):
# its Python is already installed and nothing here comes from it, so its source is dropped before the update, apt gives
# up on a dead mirror after 20 s, and the update is skipped altogether when the packages are already there.
APT_OPTS="-o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20"
# 2026-10-22: one update took ~90 s on a 4090 host (9 s on another). Every package here is in Ubuntu's own archive, so the
# update reads /etc/apt/sources.list only (not the image's NVIDIA CUDA repo in sources.list.d) when that file has entries.
apt_update() {
  if grep -qE '^[[:space:]]*deb[[:space:]]' /etc/apt/sources.list 2>/dev/null; then
    apt-get $APT_OPTS -o Dir::Etc::sourceparts=- -o APT::Get::List-Cleanup=0 update -qq
  else
    apt-get $APT_OPTS update -qq
  fi
}
apt_base() {
  dpkg -s $APT_PKGS >/dev/null 2>&1 && { log "apt: packages already installed"; return 0; }
  rm -f /etc/apt/sources.list.d/*deadsnakes* 2>/dev/null || true
  local t0=$(date +%s)
  apt_update && apt-get $APT_OPTS install -y -qq --no-install-recommends $APT_PKGS > /dev/null || return 1
  log "apt: $APT_PKGS in $(( $(date +%s) - t0 ))s"
}
# apt next to the installs (the probe: 9 s while uv installs): started before the features; waited for by the features
# that load the GL / EGL / OpenMP libraries (open3d, room, check) and at the end, never by base itself
APT_PID=""
apt_bg() {
  rm -f "$STATE/apt.rc"
  ( apt_base > "$STATE/apt.log" 2>&1; echo $? > "$STATE/apt.rc" ) &
  APT_PID=$!
}
apt_wait() {
  if [ -z "$APT_PID" ]; then apt_base; return $?; fi
  wait "$APT_PID" 2>/dev/null
  APT_PID=""
  cat "$STATE/apt.log" 2>/dev/null
  [ "$(cat "$STATE/apt.rc" 2>/dev/null)" = 0 ]
}
# one more apt install (the lists from apt_base when they exist, else a fresh update), retried once after 10 s
apt_get() {
  rm -f /etc/apt/sources.list.d/*deadsnakes* 2>/dev/null || true
  local i
  for i in 1 2; do
    apt-get $APT_OPTS install -y -qq --no-install-recommends "$@" > /dev/null 2>&1 && return 0
    apt_update > /dev/null 2>&1
    apt-get $APT_OPTS install -y -qq --no-install-recommends "$@" > /dev/null && return 0
    [ $i = 1 ] && { log "apt: install $* failed; retrying in 10 s"; sleep 10; }
  done
  return 1
}

# static ffmpeg: one tarball, two binaries, no apt and no PyPI (a few seconds from GitHub's CDN). BtbN's n8.1 build matches
# the local ffmpeg 8.1 (showinfo pts_time to 6 decimals, -fps_mode); the release-branch name moves on, hence the fallbacks.
FFMPEG_URLS=${DANCECAP_FFMPEG_URLS:-"https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n8.1-latest-linux64-gpl-8.1.tar.xz https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"}
FFMPEG_DIR=$STATE/ffmpeg_static
FFMPEG_BIN=${DANCECAP_FFMPEG_BIN:-/usr/local/bin}   # ahead of /usr/bin on PATH (tests point it elsewhere)
# One deadline for the whole static fetch, and curl drops a URL that crawls (2026-10-21 review: --max-time restarts on each
# --retry, so the 150 MB BtbN tarball alone could burn ~6 min on a slow link before the 42 MB build was tried, and a failed
# background fetch was repeated in the foreground): under FFMPEG_MIN_BPS for 15 s -> the next URL; past the deadline ->
# apt. The pid file lets a retried bootstrap (the runner reruns it at once after a quick base failure) wait for the first
# run's fetch instead of starting a second one into the same directory; the .failed marker sends a retry straight to apt.
FFMPEG_FETCH_S=${DANCECAP_FFMPEG_FETCH_S:-150}
FFMPEG_MIN_BPS=${DANCECAP_FFMPEG_MIN_BPS:-1000000}
FFMPEG_PIDF=$STATE/ffmpeg_fetch.pid
FFMPEG_FAILED=$STATE/ffmpeg_fetch.failed
have_ffmpeg() { command -v ffmpeg > /dev/null && command -v ffprobe > /dev/null && ffprobe -version > /dev/null 2>&1; }
ffmpeg_fetching() { local p; p=$(cat "$FFMPEG_PIDF" 2>/dev/null) && [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
ffmpeg_fetch() {  # -> $FFMPEG_DIR/{ffmpeg,ffprobe}; python's lzma extracts (the image may lack xz)
  local u left part=$FFMPEG_DIR.part.${BASHPID:-$$} t0=$(date +%s)
  rm -f "$FFMPEG_FAILED"
  for u in $FFMPEG_URLS; do
    left=$(( FFMPEG_FETCH_S - ($(date +%s) - t0) ))
    [ $left -lt 5 ] && { log "ffmpeg: no static build within ${FFMPEG_FETCH_S} s"; break; }
    rm -rf "$part"; mkdir -p "$part"
    if curl -fsSL --connect-timeout 15 --max-time $left --speed-limit $FFMPEG_MIN_BPS --speed-time 15 \
         -o "$part/dl.tar.xz" "$u" &&
       "$PY" - "$part/dl.tar.xz" "$part" <<'EOF'
import os, sys, tarfile
src, dst = sys.argv[1:3]
got = set()
with tarfile.open(src, "r:xz") as t:
    for m in t:
        b = os.path.basename(m.name)
        if m.isfile() and b in ("ffmpeg", "ffprobe") and b not in got:
            with t.extractfile(m) as f, open(os.path.join(dst, b), "wb") as o:
                o.write(f.read())
            os.chmod(os.path.join(dst, b), 0o755)
            got.add(b)
sys.exit(0 if len(got) == 2 else 1)
EOF
    then
      rm -f "$part/dl.tar.xz"; echo "$u" > "$part/source"; rm -rf "$FFMPEG_DIR"; mv "$part" "$FFMPEG_DIR"
      log "ffmpeg: static build fetched in $(( $(date +%s) - t0 ))s from $u"
      return 0
    fi
    log "ffmpeg: static build from $u failed after $(( $(date +%s) - t0 ))s"
  done
  rm -rf "$part"; touch "$FFMPEG_FAILED"
  return 1
}
# the background fetch, at most one per pod: a fetch still running from an earlier bootstrap run is left to finish
ffmpeg_fetch_bg() {
  if ffmpeg_fetching; then log "ffmpeg: the static fetch of an earlier bootstrap run is still going"; return 0; fi
  ( echo "${BASHPID:-$$}" > "$FFMPEG_PIDF"; ffmpeg_fetch; rc=$?; rm -f "$FFMPEG_PIDF"; exit $rc ) \
    > "$STATE/ffmpeg_fetch.log" 2>&1 &
  FFMPEG_FETCH_PID=$!
}

feature_ffmpeg() {
  if ! have_ffmpeg; then
    if [ -n "${FFMPEG_FETCH_PID:-}" ]; then
      wait "$FFMPEG_FETCH_PID"; cat "$STATE/ffmpeg_fetch.log" 2>/dev/null
    elif ffmpeg_fetching; then   # an earlier run's fetch (not our child): bounded by its own deadline
      local i=0
      while ffmpeg_fetching && [ $i -lt $(( FFMPEG_FETCH_S + 30 )) ]; do sleep 1; i=$(( i + 1 )); done
      cat "$STATE/ffmpeg_fetch.log" 2>/dev/null
    fi
    # a foreground fetch only when no fetch ran at all (a failed one goes straight to apt)
    [ -x "$FFMPEG_DIR/ffmpeg" ] || [ -f "$FFMPEG_FAILED" ] || [ -n "${FFMPEG_FETCH_PID:-}" ] || ffmpeg_fetch
    # over the image's python-package copies (imageio_ffmpeg links, the trial's front job had one without ffprobe)
    [ -x "$FFMPEG_DIR/ffmpeg" ] && [ -x "$FFMPEG_DIR/ffprobe" ] && mkdir -p "$FFMPEG_BIN" &&
      install -m 755 "$FFMPEG_DIR/ffmpeg" "$FFMPEG_DIR/ffprobe" "$FFMPEG_BIN/"
  fi
  if ! have_ffmpeg; then
    log "ffmpeg: no static build reachable; Ubuntu's apt ffmpeg"
    apt_get ffmpeg || return 1
  fi
  have_ffmpeg || return 1
  log "ffmpeg: $(command -v ffmpeg): $(ffmpeg -hide_banner -version | head -1)"
}

feature_open3d() {
  apt_wait || { log "apt failed"; return 1; }   # open3d loads libGL / libgomp
  # physics_body / room_mesh import open3d; open3d -> dash -> flask -> blinker>=1.6, and the image's distutils blinker 1.4
  # cannot be uninstalled by pip (2026-10-21 trial: 2 analyses sessions lost to this). numpy / torch held by constraints.
  local c; c=$(_constraints)
  pipq --ignore-installed "blinker>=1.6" || return 1
  pip_pinned -c "$c" open3d==0.19.0 || return 1
  "$PY" -c "import open3d; print('[bootstrap] open3d', open3d.__version__)" || return 1
}

# each feature's PyPI pins: the feature installs them itself, and the prefetch installs every requested feature's at once
PINS_base=(numpy==1.26.4 scipy==1.17.1 opencv-python-headless==4.11.0.86 PyYAML==6.0.3 tqdm pillow matplotlib==3.11.1
           smplx==0.1.28 trimesh==5.1.0 ninja jaxtyping rich)
PINS_mamma=(pandas plotly==7.0.0 scenepic==1.1.2 pyrender==0.1.45 PyOpenGL==3.1.0 kornia==0.8.3 imageio-ffmpeg scikit-learn
            loguru einops munkres yacs hydra-core psutil "setuptools<81")
PINS_open3d=("blinker>=1.6" open3d==0.19.0)
PINS_room=("blinker>=1.6" open3d==0.19.0 xatlas==0.0.11 plyfile==1.1)
PINS_gsplat=(plyfile==1.1)
PINS_dancers=("hydra-core>=1.3.2" "iopath>=0.1.10")
PINS_cameras=(pycolmap==4.2.1 kornia==0.8.3)
PINS_ma2d=(pytorch-lightning==2.0.0 torchmetrics==1.9.0 lightning-utilities==0.15.3 omegaconf==2.3.1 python-dotenv==1.2.3
           loguru==0.7.3 webdataset==1.0.2 wandb==0.29.0 scikit-image==0.26.0 albumentations==1.4.24 albucore==0.0.23
           librosa==0.11.0 yacs==0.1.8 einops==0.8.2 kornia==0.8.3 huggingface_hub nvidia-ml-py lz4 zstandard)

# ONE uv call for the pins of every requested feature not done yet (+ the torch triple with base): uv resolves and
# downloads them all at once. Advisory: when it fails, each feature installs its own pins as before.
prefetch() {
  local f specs=() t0 rc todo=() name
  for f in "$@"; do
    { [ -f "$STATE/$f.ok" ] && grep -qx "v$BOOTSTRAP_VERSION" "$STATE/$f.ok"; } && continue
    todo+=("$f")
    [ "$f" = base ] && specs+=("${TORCH_PINS[@]}")
    declare -p "PINS_$f" > /dev/null 2>&1 || continue   # ffmpeg, texture, check: nothing from PyPI
    name="PINS_$f[@]"
    specs+=("${!name}")
  done
  [ ${#specs[@]} -eq 0 ] && return 0
  uv_ready || { log "prefetch: no uv; each feature installs its own pins with pip"; return 0; }
  t0=$(date +%s)
  "$UV" pip install -q "${UVT[@]}" --extra-index-url "$TORCH_IDX" --index-strategy unsafe-best-match "${specs[@]}" \
    > "$STATE/prefetch.log" 2>&1
  rc=$?
  printf '%s\t%s\t%s\n' prefetch "$(( $(date +%s) - t0 ))" "$rc" >> "$STATE/timings.tsv"
  if [ $rc -eq 0 ]; then log "prefetch: ${#specs[@]} pins of [${todo[*]}] in $(( $(date +%s) - t0 ))s ($("$UV" --version))"
  else log "prefetch: uv failed after $(( $(date +%s) - t0 ))s, each feature installs its own pins: $(tail -3 "$STATE/prefetch.log" | tr '\n' ' ')"; fi
  return 0
}

feature_base() {
  if ! torch_ok; then
    pipq "${TORCH_PINS[@]}" --index-url "$TORCH_IDX" || return 1
  fi
  pip_pinned "${PINS_base[@]}" || return 1
  if [ "${DANCECAP_ALLOW_NO_GPU:-0}" = 1 ] || cpu_pod; then  # farm env build on a CPU pod (hm_env.sh), a runner CPU pod
    "$PY" -c "import torch; print('[bootstrap] torch', torch.__version__, 'cuda', torch.version.cuda, '(no GPU check)')"
    return
  fi
  "$PY" -c "import torch; assert torch.cuda.is_available(); print('[bootstrap] torch', torch.__version__, torch.cuda.get_device_name(0))"
}

install_wheel_or_build() {  # $1 = dist name in wheel filename, $2 = pip spec to build from
  local name=$1 spec=$2 whl
  if cpu_pod; then log "$name: a CUDA extension, not on a CPU pod"; return 1; fi
  wait_bundle   # the cached wheels come with the bundle
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
  pip_pinned "${PINS_mamma[@]}" || return 1
  if cpu_pod; then log "pytorch_sdf: skipped on a CPU pod (CUDA extension; the analyses do not use it)"; return 0; fi
  install_wheel_or_build pytorch_sdf "git+https://github.com/cuevhv/pytorch_sdf@torch2.5-cu124" || return 1
  "$PY" -c "from pytorch_sdf import sdf; print('[bootstrap] pytorch_sdf ok')"
}

feature_room() {
  apt_wait || { log "apt failed"; return 1; }
  # open3d -> dash -> flask -> blinker>=1.6, but Ubuntu ships a distutils blinker 1.4 pip cannot uninstall
  pipq --ignore-installed "blinker>=1.6" || return 1
  pip_pinned open3d==0.19.0 xatlas==0.0.11 plyfile==1.1 || return 1
  install_wheel_or_build gsplat "--no-binary gsplat gsplat==1.5.3" || return 1
  "$PY" -c "import gsplat; from gsplat.cuda._backend import _C; assert _C is not None; print('[bootstrap] gsplat', gsplat.__version__, 'CUDA ext ok')"
}

feature_texture() { return 0; }

feature_gsplat() {
  pip_pinned plyfile==1.1 || return 1
  install_wheel_or_build gsplat "--no-binary gsplat gsplat==1.5.3" || return 1
  "$PY" -c "import gsplat, plyfile; from gsplat.cuda._backend import _C; assert _C is not None; print('[bootstrap] gsplat', gsplat.__version__, 'CUDA ext ok')"
}

feature_dancers() {
  if cpu_pod; then log "dancers: needs a GPU"; return 1; fi
  pipq "hydra-core>=1.3.2" "iopath>=0.1.10" || return 1
  SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=1 pipq --no-deps --no-build-isolation       "git+https://github.com/facebookresearch/sam2@2b90b9f5ceec907a1c18123530e92e794ad901a4" || return 1
  "$PY" -c "import sam2; from sam2.build_sam import build_sam2_video_predictor; print('[bootstrap] sam2 ok', sam2.__file__)"
}

# numpy/torch must never move: every later pip install in these features runs under this constraints file
_constraints() {
  printf '%s\n' numpy==1.26.4 "torch==2.5.1$TV" "torchvision==0.20.1$TV" > "$STATE/constraints.txt"
  echo "$STATE/constraints.txt"
}

feature_cameras() {
  local c; c=$(_constraints)
  pipq -c "$c" pycolmap==4.2.1 kornia==0.8.3 || return 1
  pipq -c "$c" --no-deps "git+https://github.com/cvg/LightGlue@eb42fee2d71449efb0aa5c10549752b5d75384d8" || return 1
  "$PY" -c "import pycolmap, lightglue; from lightglue import ALIKED, LightGlue; print('[bootstrap] pycolmap', pycolmap.__version__, 'lightglue ok')"
}

feature_ma2d() {
  local c; c=$(_constraints)
  pipq -c "$c" pytorch-lightning==2.0.0 torchmetrics==1.9.0 lightning-utilities==0.15.3 omegaconf==2.3.1 \
      python-dotenv==1.2.3 loguru==0.7.3 webdataset==1.0.2 wandb==0.29.0 scikit-image==0.26.0 albumentations==1.4.24 \
      albucore==0.0.23 librosa==0.11.0 yacs==0.1.8 einops==0.8.2 kornia==0.8.3 huggingface_hub nvidia-ml-py lz4 \
      zstandard || return 1
  "$PY" -c "import pytorch_lightning, albumentations, omegaconf, kornia, skimage, librosa, webdataset; print('[bootstrap] ma2d stack ok')"
}

feature_check() {
  apt_wait || log "apt failed (the EGL / GL checks will say what is missing)"
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
def ffm():
    import subprocess
    v = subprocess.run(["ffmpeg", "-hide_banner", "-version"], capture_output=True, text=True, timeout=30).stdout
    subprocess.run(["ffprobe", "-hide_banner", "-version"], capture_output=True, check=True, timeout=30)
    return {"ffmpeg": v.splitlines()[0] if v else "?"}
def pyr():
    import pyrender  # ma_3d imports it (never renders with --skip_scene_videos)
    return {"pyrender": pyrender.__version__}
def egl():
    import pyrender
    r = pyrender.OffscreenRenderer(64, 64); r.delete()
    return "egl offscreen ok"
for n, f in [("torch_cuda", torch_cuda), ("pytorch_sdf", sdf), ("gsplat", gs), ("smplx", smplx_),
             ("open3d", o3d), ("pyrender", pyr), ("egl_offscreen", egl), ("ffmpeg", ffm)]:
    chk(n, f)
OPTIONAL = {"egl_offscreen"}  # informational: no stage renders with EGL on the pod today
import os
feats = set(os.environ.get("DANCECAP_FEATURES", "").split())
feats |= {f[:-3] for f in os.listdir(os.path.dirname(sys.argv[1])) if f.endswith(".ok")}  # installed earlier
if "mamma" not in feats:
    OPTIONAL |= {"pytorch_sdf", "pyrender", "pandas"}
if "room" not in feats:
    OPTIONAL |= {"xatlas"} | ({"open3d"} if "open3d" not in feats else set())
    if "gsplat" not in feats:
        OPTIONAL |= {"gsplat", "plyfile"}
if "ffmpeg" not in feats:
    OPTIONAL |= {"ffmpeg"}
if os.environ.get("DANCECAP_CPU_POD") == "1":   # a CPU pod: no CUDA, no EGL device
    OPTIONAL |= {"torch_cuda", "pytorch_sdf", "gsplat", "egl_offscreen"}
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

# --no-base: every job of the session brings its own environment (remote_jobs JobSpec.base = false); only the light
# features given (ffmpeg, open3d) run, on the image's python as it is
if [ "${1:-}" = --no-base ]; then shift; features=("$@"); else features=("base" "$@"); fi
export DANCECAP_FEATURES="${features[*]}"
# the static ffmpeg download needs nothing from base: it runs next to base's apt / torch / pip work (~1.5-2.5 min) and
# feature_ffmpeg only waits for it
case " ${features[*]} " in
  *" ffmpeg "*)
    if ! have_ffmpeg && [ ! -x "$FFMPEG_DIR/ffmpeg" ] && [ ! -f "$FFMPEG_FAILED" ] &&
       ! { [ -f "$STATE/ffmpeg.ok" ] && grep -qx "v$BOOTSTRAP_VERSION" "$STATE/ffmpeg.ok"; }; then
      ffmpeg_fetch_bg
    fi;;
esac
# apt next to the installs; then one uv call for the pins of every requested feature
case " ${features[*]} " in
  *" base "*) { [ -f "$STATE/base.ok" ] && grep -qx "v$BOOTSTRAP_VERSION" "$STATE/base.ok"; } || apt_bg;;
esac
prefetch "${features[@]}"
seen=" "
rc=0
for f in "${features[@]}"; do
  case "$seen" in *" $f "*) continue;; esac
  seen="$seen$f "
  if ! declare -F "feature_$f" > /dev/null; then log "unknown feature $f"; rc=2; continue; fi
  if [ "$f" = check ]; then rm -f "$STATE/check.ok"; fi  # checks always re-run
  run_feature "$f" || { rc=1; [ "$f" = base ] && break; }
done
if [ -n "$APT_PID" ]; then apt_wait || { log "apt failed"; rc=1; }; fi   # nothing waited for it yet: before the jobs do
log "finished rc=$rc (MAX_JOBS=$MAX_JOBS); wheels in $WHEELHOUSE: $(ls "$WHEELHOUSE" 2>/dev/null | tr '\n' ' ')"
exit $rc
