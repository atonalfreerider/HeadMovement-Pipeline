#!/usr/bin/env bash
# Pod setup for the face job (dancecap.face_pod), run after remote/bootstrap.sh base (torch 2.5.1+cu124,
# numpy 1.26.4, smplx 0.1.28, opencv-python-headless). Idempotent: marker $STATE/face_<part>.ok.
#
#   bash remote/face_setup.sh mediapipe   mediapipe 1.1.0 (the local .venv-face version, Apache-2.0) + the MediaPipe
#                                         Face Landmarker model face_landmarker.task (float16 v1, ~3.6 MB, Apache-2.0)
#                                         from storage.googleapis.com/mediapipe-models into dancecap/weights/face/
#                                         (the path dancecap.face_worker expects); cv2 is repaired afterwards if
#                                         mediapipe's opencv-contrib wheel clobbered it (remote/hands_setup.sh fixcv)
#   bash remote/face_setup.sh ncpu        CPUs of this container (cgroup quota), for the MediaPipe shards
#
# No gated model (MANO / FLAME) is used on the pod: the FLAME eyelid blendshapes are added locally by
# `python -m dancecap.face_expr <take> --review-only` after the outputs come back.
set -uo pipefail
WS=${DANCECAP_WS:-/workspace/HeadMovement}
STATE=${DANCECAP_BOOTSTRAP_STATE:-/workspace/.dancecap_bootstrap}
PY=${PY:-python}
W=$WS/dancecap/weights/face
MP_URL=https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_ROOT_USER_ACTION=ignore PIP_NO_INPUT=1
mkdir -p "$STATE" "$W"
log() { echo "[face_setup] $*"; }

part_mediapipe() {
  # pinned, no unpinned fallback: a different MediaPipe would change the landmarks the local runs were checked on
  "$PY" -m pip install -q mediapipe==1.1.0 numpy==1.26.4 || return 1
  if [ ! -s "$W/face_landmarker.task" ]; then
    curl -fL --retry 4 --retry-delay 5 -o "$W/face_landmarker.task.part" "$MP_URL" || return 1
    mv "$W/face_landmarker.task.part" "$W/face_landmarker.task"
    log "downloaded face_landmarker.task $(stat -c %s "$W/face_landmarker.task") B from $MP_URL"
  fi
  bash "$WS/dancecap/remote/hands_setup.sh" fixcv || true
  "$PY" -c "import mediapipe as m, cv2; print('[face_setup] mediapipe', m.__version__, 'cv2', cv2.__version__)" || return 1
}

part_ncpu() {
  bash "$WS/dancecap/remote/hands_setup.sh" ncpu
}

case "${1:-}" in
  mediapipe)
    if [ -f "$STATE/face_mediapipe.ok" ]; then log "mediapipe already set up"; exit 0; fi
    part_mediapipe && touch "$STATE/face_mediapipe.ok" ;;
  ncpu) part_ncpu ;;
  *) echo "usage: $0 mediapipe|ncpu"; exit 2 ;;
esac
