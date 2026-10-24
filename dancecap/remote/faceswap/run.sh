#!/usr/bin/env bash
# The face swap's pod commands (remote/faceswap/pod_job_template.json): `bash remote/faceswap/run.sh <film_faceswap args>`, from the
# job's cwd (dancecap/). The FaceFusion code comes from the job's shipped `git archive` of the pinned commit (FACEFUSION_ARCHIVE,
# FACEFUSION_COMMIT), extracted once to FACEFUSION_ROOT; the models install.sh fetched are linked in as its .assets/models; then
# the venv python (install.sh's ffpy: the pip CUDA libraries on LD_LIBRARY_PATH) runs `python -m dancecap.film_faceswap "$@"`.
set -uo pipefail
FF_HOME=${FF_HOME:-/workspace/ff}
FFR=${FACEFUSION_ROOT:-$FF_HOME/facefusion}
log() { echo "[ff-run] $*"; }
[ -x "$FF_HOME/ffpy" ] && [ -f "$FF_HOME/install.ok" ] || { log "$FF_HOME/ffpy missing: the install command failed"; exit 3; }
if [ "$(cat "$FFR/.hm_commit" 2>/dev/null)" != "${FACEFUSION_COMMIT:-unset}" ]; then
  [ -f "${FACEFUSION_ARCHIVE:-}" ] || { log "FaceFusion archive '${FACEFUSION_ARCHIVE:-}' not found"; exit 2; }
  mkdir -p "$FFR"
  tar -xzf "$FACEFUSION_ARCHIVE" -C "$FFR" || { log "could not extract $FACEFUSION_ARCHIVE"; exit 2; }
  echo "${FACEFUSION_COMMIT:-unset}" > "$FFR/.hm_commit"
  log "FaceFusion ${FACEFUSION_COMMIT:-?} extracted to $FFR ($(find "$FFR" -type f | wc -l) files)"
  want=$(grep -v '^onnxruntime' "$FFR/requirements.txt" | tr -d '\r' | sed '/^[[:space:]]*$/d' | sort)
  have=$(tr ' ' '\n' < "$FF_HOME/pins.txt" | sed '/^[[:space:]]*$/d' | sort)
  if [ "$want" = "$have" ]; then log "venv pins = the code's requirements.txt (onnxruntime aside)"
  else log "WARNING: the venv pins differ from the code's requirements.txt:"; diff <(echo "$want") <(echo "$have") || true; fi
fi
mkdir -p "$FFR/.assets"
if [ -d "$FF_HOME/models" ] && [ ! -e "$FFR/.assets/models" ]; then ln -s "$FF_HOME/models" "$FFR/.assets/models"; fi
[ -f "$FF_HOME/models.log" ] && [ ! -f "$FF_HOME/models.shown" ] && { tail -1 "$FF_HOME/models.log"; touch "$FF_HOME/models.shown"; }
exec "$FF_HOME/ffpy" -m dancecap.film_faceswap "$@"
