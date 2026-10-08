#!/usr/bin/env bash
# Prepare a pod's LOCAL disk from the volume (the volume is a network FS: tar in, tar out, never thousands of small files).
#
#   pod_prepare.sh [env] [code] [libs] [links]     default: all four, in that order
#   pod_prepare.sh pack-env                         pack /opt/hm/env -> $HM/env/env-<id>.tar (+ CURRENT), used by hm_env.sh
#
#   env    $HM/env/CURRENT (tar name + sha256) -> /opt/hm/env, skipped when the local copy has the same id
#   code   $HM/code/CURRENT (key) -> $HM/code/<key>.tar.gz -> /opt/hm/ws/{dancecap,mamma-turing}   (a few seconds)
#   libs   cached .debs (EGL/GL for pyrender, ffmpeg 4.4, pigz) via dpkg; apt-get as the fallback
#   links  /opt/hm/ws/mamma/data -> $HM/models/mamma_data, /opt/hm/ws/smplx -> $HM/models/smplx,
#          dancecap/weights -> $HM/weights/dancecap, dancecap/work -> /opt/hm/work (local scratch)
set -uo pipefail
HM=${HM_ROOT:-${HM:-/workspace/hm}}
ENVD=${HM_ENV_LOCAL:-/opt/hm/env}
WS=${HM_WS:-/opt/hm/ws}
LOCAL=$(dirname "$ENVD")
log() { echo "[prepare $(date -u +%T)] $*"; }
ENV_ID_FILE=$ENVD/.pack_id

unpack_env() {
  [ -f "$HM/env/CURRENT" ] || { log "env: no $HM/env/CURRENT (run hm_env.sh)"; return 1; }
  read -r name sha < "$HM/env/CURRENT"
  if [ -x "$ENVD/bin/python" ] && [ "$(cat "$ENV_ID_FILE" 2>/dev/null)" = "$sha" ]; then log "env: local copy is current ($name)"; return 0; fi
  local t0=$(date +%s)
  rm -rf "$ENVD"; mkdir -p "$LOCAL"
  tar -xf "$HM/env/$name" -C "$LOCAL" || { log "env: extract failed"; return 1; }
  echo "$sha" > "$ENV_ID_FILE"
  log "env: unpacked $name in $(( $(date +%s) - t0 )) s ($(du -sh "$ENVD" | cut -f1))"
}

pack_env() {
  [ -x "$ENVD/bin/python" ] || { log "pack: no env at $ENVD"; return 1; }
  local id; id=$(date -u +%Y%m%d-%H%M%S)
  local t0=$(date +%s)
  mkdir -p "$HM/env"
  rm -f "$ENV_ID_FILE"
  tar -cf "$HM/env/env-$id.tar.part" -C "$LOCAL" "$(basename "$ENVD")" || return 1
  mv "$HM/env/env-$id.tar.part" "$HM/env/env-$id.tar"
  local sha; sha=$(sha256sum "$HM/env/env-$id.tar" | cut -d' ' -f1)
  echo "env-$id.tar $sha" > "$HM/env/CURRENT.tmp" && mv "$HM/env/CURRENT.tmp" "$HM/env/CURRENT"
  echo "$sha" > "$ENV_ID_FILE"
  log "pack: env-$id.tar $(du -h "$HM/env/env-$id.tar" | cut -f1) sha256 ${sha:0:12} in $(( $(date +%s) - t0 )) s"
  ls -t "$HM"/env/env-*.tar 2>/dev/null | tail -n +3 | xargs -r rm -f   # keep the two newest
}

unpack_code() {
  [ -f "$HM/code/CURRENT" ] || { log "code: no $HM/code/CURRENT"; return 1; }
  local key; key=$(cat "$HM/code/CURRENT")
  if [ "$(cat "$WS/.code_key" 2>/dev/null)" = "$key" ]; then log "code: $key already unpacked"; return 0; fi
  local t0=$(date +%s)
  mkdir -p "$WS"
  rm -rf "$WS/dancecap/dancecap" "$WS/dancecap/takes" "$WS/dancecap/remote" "$WS/mamma-turing"
  tar -xzf "$HM/code/$key.tar.gz" -C "$WS" || { log "code: extract failed"; return 1; }
  echo "$key" > "$WS/.code_key"
  log "code: $key unpacked in $(( $(date +%s) - t0 )) s"
}

libs() {
  need() { ! ldconfig -p 2>/dev/null | grep -q 'libEGL.so.1' || ! command -v ffmpeg > /dev/null || ! command -v pigz > /dev/null; }
  need || { log "libs: present"; return 0; }
  local t0=$(date +%s)
  if ls "$HM"/debs/*.deb > /dev/null 2>&1; then dpkg -i "$HM"/debs/*.deb > /tmp/dpkg.log 2>&1 || log "libs: dpkg reported errors (see /tmp/dpkg.log)"; fi
  if need; then
    log "libs: cached debs insufficient, apt-get"
    (apt-get update -qq && apt-get install -y -qq --no-install-recommends libgl1 libegl1 libgles2 libglib2.0-0 libgomp1 \
       libxrender1 pigz ffmpeg > /dev/null) > /tmp/apt.log 2>&1 || log "libs: apt-get failed"
  fi
  log "libs: ready in $(( $(date +%s) - t0 )) s"
}

links() {
  mkdir -p "$WS/mamma" "$WS/dancecap" /opt/hm/work "$HM/models/mamma_data" "$HM/models/smplx" "$HM/weights/dancecap"
  ln -sfn "$HM/weights/mamma" "$HM/models/mamma_data/weights"
  ln -sfn "$HM/models/mamma_data" "$WS/mamma/data"
  ln -sfn "$HM/models/smplx" "$WS/smplx"
  ln -sfn "$HM/weights/dancecap" "$WS/dancecap/weights"
  [ -L "$WS/dancecap/work" ] || { rm -rf "$WS/dancecap/work"; ln -s /opt/hm/work "$WS/dancecap/work"; }
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(env code libs links)
rc=0
for s in "${steps[@]}"; do
  case "$s" in
    env) unpack_env || rc=1;;
    code) unpack_code || rc=1;;
    libs) libs;;
    links) links;;
    pack-env) pack_env || rc=1;;
    *) log "unknown step $s"; rc=2;;
  esac
done
exit $rc
