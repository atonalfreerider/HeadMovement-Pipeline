#!/usr/bin/env bash
# Build / refresh the farm environment ONCE and store it on the network volume as a single tar.
#
#   bash /workspace/hm/bin/hm_env.sh [feature ...]       # default: mamma dancers cameras ma2d detect texture analysis asr audio check
#   features: base mamma dancers cameras ma2d detect texture room gsplat analysis asr audio check   (room = open3d/xatlas/gsplat;
#   analysis = scipy/matplotlib/trimesh/rapidfuzz/... for stitch, timing, physics...; asr = faster-whisper; audio = beat_this)
#
# WHY A TAR: the volume is a network filesystem (MooseFS): 1.2 GB/s sequential write, 480 MB/s read, but ~24 ms per small
# file created and ~12 ms per small file read (measured 2026-10-08). A python env is ~100k small files, so a venv living
# on the volume would need many minutes to install and to import cold. The env is therefore built on the pod's LOCAL disk
# (/opt/hm/env, the same path on every pod so venv scripts stay valid) and packed into /workspace/hm/env/env-<id>.tar;
# pod_prepare.sh unpacks it onto the local disk in well under a minute.
#
# Python: the image's python3.11 (venv at /opt/hm/env). Compiled CUDA wheels (pytorch_sdf, gsplat; arch list
# 8.0;8.9;9.0 so A100/A40/L4/4090/H100 run native code) are cached as files in /workspace/hm/wheels/<key>/. The pins and
# per-feature install steps are remote/bootstrap.sh (shared with the one-pod sessions); this wrapper redirects its
# python, state and wheelhouse and adds `detect` (ultralytics, as dancecap.front_full) and a version/sha256 manifest.
set -uo pipefail
HM=${HM:-/workspace/hm}
ENVD=${HM_ENV_LOCAL:-/opt/hm/env}
WS=${HM_WS:-/opt/hm/ws}
STATE=$ENVD/.state
WHEEL_KEY=cp311-torch2.5.1-cu124-sm80_89_90
log() { echo "[hm_env $(date -u +%T)] $*"; }

mkdir -p "$HM"/{env,wheels/$WHEEL_KEY,debs,logs,bin} "$(dirname "$ENVD")" /tmp/pip-cache
# 1. start from the packed env when there is one (an update installs on top of it)
if [ ! -x "$ENVD/bin/python" ] && [ -f "$HM/env/CURRENT" ]; then
  bash "$HM/bin/pod_prepare.sh" env || true
fi
PYBIN=${PYBIN:-$(command -v python3.11 || command -v python3 || command -v python)}
log "base python: $PYBIN $($PYBIN --version 2>&1); env $ENVD"
if [ ! -x "$ENVD/bin/python" ]; then
  "$PYBIN" -m venv "$ENVD" 2>/dev/null || { "$PYBIN" -m pip install -q virtualenv && "$PYBIN" -m virtualenv -q "$ENVD"; } || { log "venv failed"; exit 2; }
fi
mkdir -p "$STATE"
"$ENVD/bin/python" -m pip install -q --upgrade pip wheel "setuptools<81" 2>&1 | tail -1

export PY=$ENVD/bin/python PATH=$ENVD/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export DANCECAP_WS=$WS DANCECAP_BOOTSTRAP_STATE=$STATE DANCECAP_WHEELHOUSE=$HM/wheels/$WHEEL_KEY
export DANCECAP_PYWHEELS=/nonexistent       # PyPI only (the pod's link is fast; the local pywheels are for slow hosts)
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-"8.0;8.9;9.0"}
export PIP_CACHE_DIR=/tmp/pip-cache TMPDIR=/tmp
# bootstrap.sh installs with uv: its cache on the env's local disk (hardlinks), never on the volume (24 ms per small file)
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}
[ -z "$(nvidia-smi -L 2>/dev/null)" ] && export DANCECAP_ALLOW_NO_GPU=1   # CPU build pod: the GPU assert in remote/bootstrap.sh base is skipped

FEATURES=("$@")
[ ${#FEATURES[@]} -eq 0 ] && FEATURES=(mamma dancers cameras ma2d detect texture analysis asr audio check)
WANT_DETECT=0; WANT_CHECK=0; EXTRA=(); BOOT=()
for f in "${FEATURES[@]}"; do
  case "$f" in detect) WANT_DETECT=1;; check) WANT_CHECK=1;; analysis|asr|audio) EXTRA+=("$f");; *) BOOT+=("$f");; esac
done

rc=0
if [ ${#BOOT[@]} -gt 0 ]; then
  bash "$WS/dancecap/remote/bootstrap.sh" "${BOOT[@]}" || rc=$?
fi

if [ "$WANT_DETECT" = 1 ]; then
  if [ -f "$STATE/detect.ok" ] && grep -qx v1 "$STATE/detect.ok"; then
    log "detect: already done"
  else
    t0=$(date +%s); drc=0
    printf '%s\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 > "$STATE/constraints.txt"
    "$PY" -m pip install -q -c "$STATE/constraints.txt" ultralytics==8.4.138 opencv-python==4.11.0.86 numpy==1.26.4 || drc=1
    if [ $drc = 0 ]; then
      "$PY" -c "import ultralytics, cv2, torch; print('[hm_env] ultralytics', ultralytics.__version__, 'cv2', cv2.__version__, 'torch', torch.__version__)" \
        && echo v1 > "$STATE/detect.ok" || drc=1
    fi
    printf 'detect\t%s\t%s\n' "$(( $(date +%s) - t0 ))" "$drc" >> "$STATE/timings.tsv"
    [ $drc != 0 ] && { log "detect: FAILED"; rc=1; }
  fi
fi


# extra pip features (not in remote/bootstrap.sh): every one runs under the constraints file so numpy / torch never move
pip_extra() {  # pip_extra <feature> <version tag> <import check> <pip args...>
  local f=$1 v=$2 chk=$3; shift 3
  if [ -f "$STATE/$f.ok" ] && grep -qx "$v" "$STATE/$f.ok"; then log "$f: already done"; return 0; fi
  local t0=$(date +%s) frc=0
  printf '%s\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 > "$STATE/constraints.txt"
  "$PY" -m pip install -q -c "$STATE/constraints.txt" "$@" || frc=1
  [ $frc = 0 ] && { "$PY" -c "$chk" || frc=1; }
  [ $frc = 0 ] && echo "$v" > "$STATE/$f.ok"
  printf '%s\t%s\t%s\n' "$f" "$(( $(date +%s) - t0 ))" "$frc" >> "$STATE/timings.tsv"
  [ $frc != 0 ] && { log "$f: FAILED"; return 1; }
  log "$f: done in $(( $(date +%s) - t0 )) s"
}
for f in "${EXTRA[@]}"; do
  case "$f" in
    # everything the analysis modules import that bootstrap.sh base/mamma/room do not: stitch, pose_quality, timing,
    # physics(_body/_spline), counterbalance, floorcraft, analysis_review, mamma_seams, moves/* (rapidfuzz), judgments
    # (typesafe-sdk: imports only; no key ever goes to a pod, LLM judgments stay on the PC)
    analysis) pip_extra analysis v1 "import scipy, matplotlib, trimesh, cv2, yaml, rapidfuzz, typesafe_sdk, pandas, librosa, soxr; print('[hm_env] analysis imports ok')" \
                rapidfuzz typesafe-sdk==0.7.2 scipy==1.17.1 matplotlib==3.11.1 trimesh==5.1.0 PyYAML pandas soundfile || rc=1;;
    # narration ASR (dancecap/moves/annotate/asr_worker.py): the same pins as the PC .venv-asr
    asr) pip_extra asr v1 "import faster_whisper, ctranslate2; print('[hm_env] faster-whisper', faster_whisper.__version__, 'ctranslate2', ctranslate2.__version__)" \
                faster-whisper==1.2.1 ctranslate2==4.8.2 || rc=1;;
    # beat grid (beats.py / song.py): the user beat_this fork (when pushed), else the upstream package
    audio) pip_extra audio v1 "import beat_this, librosa, soxr; print('[hm_env] beat_this ok')" "git+https://github.com/atonalfreerider/beat_this.git" \
             || pip_extra audio v1 "import beat_this; print('[hm_env] beat_this upstream ok')" "git+https://github.com/CPJKU/beat_this.git" || rc=1;;
  esac
done

if [ "$WANT_CHECK" = 1 ]; then
  bash "$WS/dancecap/remote/bootstrap.sh" check || rc=$?
fi

# system libs as .debs for pod_prepare.sh (offline dpkg at pod start): apt downloads the closure of what is missing now
if [ ! -f "$HM/debs/.done" ]; then
  log "caching system .debs"
  ( apt-get update -qq && apt-get install -y -qq --download-only --no-install-recommends libgl1 libegl1 libgles2 \
      libglib2.0-0 libgomp1 libxrender1 pigz ffmpeg > /dev/null 2>&1 ; \
    cp -n /var/cache/apt/archives/*.deb "$HM/debs/" 2>/dev/null; touch "$HM/debs/.done" ) || true
  log "debs: $(ls "$HM/debs" | grep -c '\.deb$')"
fi

# pack: one sequential write to the volume, atomic, with a CURRENT pointer + sha256
"$PY" "$WS/dancecap/dancecap/farm/bootstrap/manifest.py" env --volume "$HM" --env-dir "$ENVD" > "$HM/logs/manifest_env.log" 2>&1 \
  || log "manifest failed (see logs/manifest_env.log)"
if [ "${HM_NO_PACK:-0}" != 1 ]; then
  bash "$HM/bin/pod_prepare.sh" pack-env || { log "pack failed"; rc=1; }
fi
log "finished rc=$rc"
exit $rc
