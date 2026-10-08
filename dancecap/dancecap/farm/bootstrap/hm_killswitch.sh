#!/usr/bin/env bash
# On-pod dead-man switch (farm): deletes THIS pod when the deadline in <deadline_file> (epoch seconds) has passed.
#
#   hm_killswitch.sh <deadline_file> [hard_ceiling_epoch]
#
# * The deadline file lives on the network volume (/workspace/hm/pods/<pod id>/kill_at), so the local side can extend
#   it over ssh while a pod is healthy; it can never be extended past the hard ceiling (given at creation).
# * A missing or unreadable file falls back to HM_KILL_AT (the creation-time deadline), never to "now": a flaky
#   volume read must not kill a working pod.
# * Runpod injects a pod-scoped RUNPOD_API_KEY and runpodctl into the container; when this runs from an ssh session
#   the variables are taken from PID 1.
set -u
DL=${1:?deadline file}
HARD=${2:-${HM_HARD_CEIL:-0}}
INIT=${HM_KILL_AT:-0}
if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
  set -a; . <(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed -E "s/^([A-Z_]+)=(.*)$/\1='\2'/"); set +a
fi
log() { echo "[killswitch $(date -u +%T)] $*"; }
log "armed: pod=${RUNPOD_POD_ID:-?} file=$DL init=$INIT hard=$HARD"
while true; do
  now=$(date +%s)
  dl=$(cat "$DL" 2>/dev/null | tr -dc '0-9')
  [ -z "$dl" ] && dl=$INIT
  [ -z "$dl" ] && dl=0
  if [ "$HARD" -gt 0 ] && [ "$dl" -gt "$HARD" ]; then dl=$HARD; fi
  if [ "$dl" -gt 0 ] && [ "$now" -ge "$dl" ]; then
    log "deadline $dl reached: deleting pod ${RUNPOD_POD_ID:-?}"
    runpodctl pod delete "$RUNPOD_POD_ID" || runpodctl remove pod "$RUNPOD_POD_ID"
    sleep 120
  fi
  sleep 20
done
