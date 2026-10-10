#!/usr/bin/env bash
# Speech environment of the hm-farm volume: the farm env (faster-whisper 1.2.1, torch 2.5.1, numpy 1.26.4 ...) plus NVIDIA NeMo (parakeet,
# TitaNet, Sortformer), speechbrain (ECAPA) and transformers (AST audio tagging). Built ONCE on a pod, stored as one tar on the volume,
# unpacked next to the farm env on every speech job: the farm env itself (env/CURRENT, used by every other job) is never touched.
#
#   speech_env.sh build     pod with the volume: unpack the farm env if needed, pip install the extras under constraints, import-test,
#                           pack /opt/hm/env -> $HM/speech/speech-env-<id>.tar (members env_speech/...) + $HM/speech/CURRENT
#   speech_env.sh unpack    /opt/hm/env_speech from CURRENT (skipped when the local copy has the same id)
#   speech_env.sh info      what is installed
#
# The relocated venv runs with `python -m ...` (its scripts keep the old shebang; nothing here uses them).
set -uo pipefail
HM=${HM_ROOT:-/workspace/hm}
LOCAL=${HM_LOCAL:-/opt/hm}
SRC_ENV=$LOCAL/env
SP_ENV=$LOCAL/env_speech
SDIR=$HM/speech
log() { echo "[speech_env $(date -u +%T)] $*"; }

build() {
  mkdir -p "$SDIR" /tmp/pip-cache
  [ -x "$SRC_ENV/bin/python" ] || bash "$HM/bin/pod_prepare.sh" env || { log "no farm env"; return 1; }
  local PY=$SRC_ENV/bin/python t0=$(date +%s)
  printf '%s\n' numpy==1.26.4 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 faster-whisper==1.2.1 ctranslate2==4.8.2 'pyarrow<20' 'datasets<4.0' > /tmp/speech_constraints.txt
  export PIP_CACHE_DIR=/tmp/pip-cache TMPDIR=/tmp
  "$PY" -m pip install -q --upgrade pip wheel "setuptools<81" 2>&1 | tail -1
  log "pip install nemo_toolkit[asr] speechbrain transformers ..."
  "$PY" -m pip install -c /tmp/speech_constraints.txt "nemo_toolkit[asr]==${NEMO_VERSION:-2.5.3}" "speechbrain>=1.0,<1.2" soundfile scikit-learn \
      > /tmp/speech_pip.log 2>&1
  local rc=$?
  tail -5 /tmp/speech_pip.log
  cp /tmp/speech_pip.log "$SDIR/pip_install.log" 2>/dev/null || true
  [ $rc = 0 ] || { log "pip install FAILED rc=$rc (log: $SDIR/pip_install.log)"; return 1; }
  "$PY" -m pip freeze > "$SDIR/pip_freeze.txt" 2>/dev/null
  "$PY" -m pip check > "$SDIR/pip_check.txt" 2>&1 || true
  log "installed in $(( $(date +%s) - t0 )) s; pip check: $(wc -l < "$SDIR/pip_check.txt") lines"
  "$PY" - <<'PYEOF' || return 1
import importlib, sys
bad = []
for m in ["numpy", "torch", "faster_whisper", "ctranslate2", "soundfile", "scipy", "librosa", "sklearn", "nemo.collections.asr",
          "speechbrain", "transformers"]:
    try:
        mod = importlib.import_module(m)
        print("ok  ", m, getattr(mod, "__version__", ""))
    except Exception as e:  # noqa: BLE001
        bad.append(m)
        print("FAIL", m, type(e).__name__, str(e)[:200])
sys.exit(1 if bad else 0)
PYEOF
  local id; id=$(date -u +%Y%m%d-%H%M%S)
  tar -cf "$SDIR/speech-env-$id.tar.part" --transform='s,^env/,env_speech/,' -C "$LOCAL" env || return 1
  mv "$SDIR/speech-env-$id.tar.part" "$SDIR/speech-env-$id.tar"
  local sha; sha=$(sha256sum "$SDIR/speech-env-$id.tar" | cut -d' ' -f1)
  echo "speech-env-$id.tar $sha" > "$SDIR/CURRENT.tmp" && mv "$SDIR/CURRENT.tmp" "$SDIR/CURRENT"
  ls -t "$SDIR"/speech-env-*.tar 2>/dev/null | tail -n +3 | xargs -r rm -f      # keep the two newest
  log "packed speech-env-$id.tar $(du -h "$SDIR/speech-env-$id.tar" | cut -f1) in $(( $(date +%s) - t0 )) s"
}

unpack() {
  [ -f "$SDIR/CURRENT" ] || { log "no $SDIR/CURRENT: build the speech env first"; return 1; }
  read -r name sha < "$SDIR/CURRENT"
  if [ -x "$SP_ENV/bin/python" ] && [ "$(cat "$SP_ENV/.speech_id" 2>/dev/null)" = "$sha" ]; then log "local copy is current ($name)"; return 0; fi
  local t0=$(date +%s)
  rm -rf "$SP_ENV"
  tar -xf "$SDIR/$name" -C "$LOCAL" || { log "extract failed"; return 1; }
  echo "$sha" > "$SP_ENV/.speech_id"
  log "unpacked $name in $(( $(date +%s) - t0 )) s ($(du -sh "$SP_ENV" | cut -f1))"
}

info() {
  local PY=$SP_ENV/bin/python; [ -x "$PY" ] || PY=$SRC_ENV/bin/python
  "$PY" -m pip list 2>/dev/null | grep -i -E "^(torch|torchaudio|numpy|faster|ctranslate2|nemo|speechbrain|transformers|lightning|hydra|omegaconf|onnx|librosa|scipy|scikit|huggingface|protobuf|numba) "
}

case "${1:-}" in
  build) build;;
  unpack) unpack;;
  info) info;;
  *) echo "usage: speech_env.sh build|unpack|info"; exit 2;;
esac
