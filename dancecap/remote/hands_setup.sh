#!/usr/bin/env bash
# Pod setup for the hands job (dancecap.hands_pod), run after remote/bootstrap.sh base (torch 2.5.1+cu124,
# numpy 1.26.4, smplx 0.1.28, opencv-python-headless, trimesh). Idempotent: markers $STATE/hands_<part>.ok.
#
#   bash remote/hands_setup.sh detectors     rtmlib 0.0.16 + onnxruntime-gpu + mediapipe 1.1.0 (the local .venv-face
#                                            versions); RTMW-x 384x288 and RTMPose-m hand ONNX from download.openmmlab.com
#                                            (Apache-2.0) and the MediaPipe hand landmarker from storage.googleapis.com
#                                            (Apache-2.0) into dancecap/weights/ (paths hand_worker expects)
#   bash remote/hands_setup.sh wilor         WiLoR (github.com/rolpotamias/WiLoR @ WILOR_COMMIT, CC-BY-NC-ND 4.0) + its
#                                            inference deps; wilor_final.ckpt + detector.pt from the authors' Hugging Face
#                                            space; the user's MANO_RIGHT.pkl (uploaded plain-numpy copy) -> mano_data/
#   bash remote/hands_setup.sh hamer_fetch   start the 6 GB HaMeR demo-data download in the background (cs.utexas.edu)
#   bash remote/hands_setup.sh hamer         HaMeR (github.com/geopavlakos/hamer @ HAMER_COMMIT, installed --no-deps: our
#                                            boxes replace its detectron2 / ViTPose front end) + checkpoint + MANO_RIGHT
#   bash remote/hands_setup.sh ncpu          CPUs of this container (cgroup quota; nproc shows the host's)
#   bash remote/hands_setup.sh fixcv         make cv2 importable again if mediapipe/ultralytics clobbered it
#
# MANO (mano.is.tue.mpg.de) is licence-restricted: it only ever lives in this private pod and is deleted with it.
set -uo pipefail
WS=${DANCECAP_WS:-/workspace/HeadMovement}
EXT=${DANCECAP_EXT:-/workspace/ext}
STATE=${DANCECAP_BOOTSTRAP_STATE:-/workspace/.dancecap_bootstrap}
PY=${PY:-python}
W=$WS/dancecap/weights
MANO_SRC=$WS/models/mano/nochumpy/MANO_RIGHT.pkl
WILOR_COMMIT=fcb911312a38fa8badd30d9656a167485d61b8f9
HAMER_COMMIT=3a01849f4148352e9260b69bf28b65d1671a4905
HAMER_URL=https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
WILOR_HF=https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models
OMM=https://download.openmmlab.com/mmpose/v1/projects
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_INPUT=1
mkdir -p "$EXT" "$STATE" "$W/face/rtmw_x384" "$W/hands"
log() { echo "[hands_setup] $*"; }
pipq() { "$PY" -m pip install -q "$@"; }
fetch() {  # url dest: resumable download; logs the size and the source
  local url=$1 dst=$2
  if [ -s "$dst" ]; then log "have $(basename "$dst") ($(stat -c %s "$dst") B)"; return 0; fi
  curl -fL --retry 4 --retry-delay 5 -C - -o "$dst.part" "$url" || return 1
  mv "$dst.part" "$dst" && log "downloaded $(basename "$dst") $(stat -c %s "$dst") B from $url"
}
# model inference deps shared by WiLoR and HaMeR (pinned where it matters; numpy stays at the base 1.26.4)
MODEL_DEPS=(pytorch-lightning==2.4.0 yacs==0.1.8 timm==0.9.16 einops scikit-image pyrender==0.1.45 PyOpenGL==3.1.0
            webdataset braceexpand pandas rich numpy==1.26.4 smplx==0.1.28)

part_detectors() {
  pipq --no-deps rtmlib==0.0.16 || return 1
  pipq onnxruntime-gpu==1.22.0 || pipq onnxruntime-gpu || return 1
  pipq mediapipe==1.1.0 numpy==1.26.4 || pipq mediapipe numpy==1.26.4 || log "WARNING: mediapipe not installed"
  local tmp=/workspace/dl
  mkdir -p $tmp
  fetch $OMM/rtmw/onnx_sdk/rtmw-dw-x-l_simcc-cocktail14_270e-384x288_20231122.zip $tmp/rtmw-x.zip || return 1
  fetch $OMM/rtmposev1/onnx_sdk/rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.zip \
        $tmp/rtmpose_m_hand5.zip || return 1
  fetch https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task \
        "$W/hands/hand_landmarker.task" || return 1
  rm -rf $tmp/rtmw && "$PY" -m zipfile -e $tmp/rtmw-x.zip $tmp/rtmw || return 1
  cp "$(find $tmp/rtmw -name end2end.onnx | head -1)" "$W/face/rtmw_x384/end2end.onnx" || return 1
  rm -rf "$W/hands/rtmpose_m_hand5" && "$PY" -m zipfile -e $tmp/rtmpose_m_hand5.zip "$W/hands/rtmpose_m_hand5" || return 1
  part_fixcv
  "$PY" -c "import rtmlib, onnxruntime as o, cv2; print('[hands_setup] rtmlib ok; ort', o.__version__, o.get_available_providers(), 'cv2', cv2.__version__)" || return 1
  "$PY" -c "import mediapipe as m; print('[hands_setup] mediapipe', m.__version__)" || log "WARNING: mediapipe import failed"
}

part_wilor() {
  if [ ! -d $EXT/WiLoR/.git ]; then git clone -q https://github.com/rolpotamias/WiLoR.git $EXT/WiLoR || return 1; fi
  git -C $EXT/WiLoR checkout -q $WILOR_COMMIT || return 1
  pipq "${MODEL_DEPS[@]}" "ultralytics==8.1.34" || return 1
  mkdir -p $EXT/WiLoR/pretrained_models $EXT/WiLoR/mano_data
  fetch $WILOR_HF/wilor_final.ckpt $EXT/WiLoR/pretrained_models/wilor_final.ckpt || return 1
  fetch $WILOR_HF/detector.pt $EXT/WiLoR/pretrained_models/detector.pt || return 1
  cp "$MANO_SRC" $EXT/WiLoR/mano_data/MANO_RIGHT.pkl || { log "MANO_RIGHT.pkl missing in the bundle"; return 1; }
  part_fixcv
  (cd $EXT/WiLoR && PYOPENGL_PLATFORM=egl "$PY" -c "import sys; sys.path.insert(0, '.'); from wilor.models import load_wilor; from wilor.datasets.vitdet_dataset import ViTDetDataset; from ultralytics import YOLO; print('[hands_setup] wilor imports ok')") || return 1
}

part_hamer_fetch() {
  mkdir -p $EXT/hamer_dl
  if [ -s $EXT/hamer_dl/hamer_demo_data.tar.gz ]; then log "hamer data already downloaded"; return 0; fi
  setsid nohup bash -c "curl -fL --retry 4 --retry-delay 5 -C - -o $EXT/hamer_dl/hamer_demo_data.tar.gz.part $HAMER_URL && mv $EXT/hamer_dl/hamer_demo_data.tar.gz.part $EXT/hamer_dl/hamer_demo_data.tar.gz; echo \$? > $EXT/hamer_dl/rc" > $EXT/hamer_dl/curl.log 2>&1 < /dev/null &
  log "hamer demo data download started in the background ($HAMER_URL)"
}

part_hamer() {
  if [ ! -d $EXT/hamer/.git ]; then git clone -q https://github.com/geopavlakos/hamer.git $EXT/hamer || return 1; fi
  git -C $EXT/hamer checkout -q $HAMER_COMMIT || return 1
  pipq "${MODEL_DEPS[@]}" || return 1
  pipq --no-deps -e $EXT/hamer || return 1
  [ -f $EXT/hamer_dl/rc ] || [ -s $EXT/hamer_dl/hamer_demo_data.tar.gz ] || part_hamer_fetch
  local n=0
  while [ ! -f $EXT/hamer_dl/rc ] && [ ! -s $EXT/hamer_dl/hamer_demo_data.tar.gz ] && [ $n -lt 180 ]; do sleep 10; n=$((n + 1)); done
  [ -s $EXT/hamer_dl/hamer_demo_data.tar.gz ] || { log "hamer data download failed: $(tail -3 $EXT/hamer_dl/curl.log)"; return 1; }
  log "hamer_demo_data.tar.gz $(stat -c %s $EXT/hamer_dl/hamer_demo_data.tar.gz) B from $HAMER_URL"
  if [ ! -f $EXT/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt ]; then
    (cd $EXT/hamer && tar --no-same-owner --warning=no-unknown-keyword --exclude=".*" -xf $EXT/hamer_dl/hamer_demo_data.tar.gz \
       --wildcards '_DATA/hamer_ckpts/*' '_DATA/data/*') || return 1
  fi
  mkdir -p $EXT/hamer/_DATA/data/mano
  cp "$MANO_SRC" $EXT/hamer/_DATA/data/mano/MANO_RIGHT.pkl || return 1
  rm -f $EXT/hamer_dl/hamer_demo_data.tar.gz
  part_fixcv
  (cd $EXT/hamer && PYOPENGL_PLATFORM=egl "$PY" -c "from hamer.models import load_hamer; from hamer.datasets.vitdet_dataset import ViTDetDataset; print('[hands_setup] hamer imports ok')") || return 1
}

part_ncpu() {
  local c q p
  c=$(nproc)
  if [ -r /sys/fs/cgroup/cpu.max ] && read -r q p < /sys/fs/cgroup/cpu.max && [ "$q" != max ]; then
    c=$(( (q + p - 1) / p ))
  elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] && q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) && [ "$q" -gt 0 ]; then
    # cgroup v1 hosts (2026-10-07: an A40 pod with nproc 96 and a 7.65-CPU quota ran the hands build 10x slower
    # with 96 OpenMP threads)
    p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us 2>/dev/null || echo 100000)
    c=$(( (q + p - 1) / p ))
  fi
  echo "$c"
}

part_fixcv() {
  if ! "$PY" -c "import cv2; cv2.imread; cv2.warpAffine" 2>/dev/null; then
    log "cv2 broken by a mixed opencv install - reinstalling opencv-contrib-python 4.11.0.86"
    "$PY" -m pip uninstall -y -q opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
    pipq --no-deps opencv-contrib-python==4.11.0.86 || return 1
  fi
  return 0
}

for part in "$@"; do
  case $part in
    ncpu|fixcv|hamer_fetch) "part_$part"; continue ;;
  esac
  if [ -f "$STATE/hands_$part.ok" ]; then log "$part: already done"; continue; fi
  t0=$(date +%s)
  if "part_$part"; then
    echo ok > "$STATE/hands_$part.ok"; log "$part: done in $(( $(date +%s) - t0 ))s"
  else
    log "$part: FAILED after $(( $(date +%s) - t0 ))s"; exit 1
  fi
done
