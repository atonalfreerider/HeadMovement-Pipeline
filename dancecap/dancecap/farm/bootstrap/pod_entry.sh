#!/usr/bin/env bash
# Container main process of a farm pod (FARM.md section 5/6): pods.create() passes it as the docker command, so a pod is
# useful the moment the container is up, with or without the local watcher:
#
#   1. Runpod's own /start.sh in the background (sshd with the account's keys)
#   2. kill switch (hm_killswitch.sh) armed from HM_KILL_AT / HM_HARD_CEIL, deadline file on the volume
#   3. pod_prepare.sh: venv tar (env/CURRENT) -> /opt/hm/env, code bundle (code/CURRENT) -> /opt/hm/ws, cached system .debs,
#      links (models/weights/scratch)  -- the volume is a network FS, so big tars in, never thousands of small files
#   4. the worker (dancecap.farm.worker): claims jobs from queue/, exits when the queue is empty for HM_IDLE_MIN minutes
#   5. after the worker exits: the deadline is pulled in to now + HM_POST_GRACE_MIN (the local watcher normally deletes the
#      pod first, after pulling; the grace only guards against a dead watcher) and the container stays up for ssh
#
# Environment (set by pods.create): HM_ROOT (/workspace/hm), HM_AUTOSTART (1), HM_KINDS, HM_MAX_JOBS, HM_IDLE_MIN,
# HM_POST_GRACE_MIN, HM_KILL_AT, HM_HARD_CEIL, HM_POD_NAME, HM_PRICE_HR. HM_AUTOSTART=0: kill switch + ssh only (gateway).
# Timeline of one start is written to $HM/pods/<id>/events.log (epoch seconds): entry, prepared, worker, first job.
set -u
HM=${HM_ROOT:-/workspace/hm}
ID=${RUNPOD_POD_ID:-local}
PD=$HM/pods/$ID
T_ENTRY=$(date +%s)
mkdir -p "$PD" 2>/dev/null
ev() { echo "$(date +%s) $*" >> "$PD/events.log"; echo "[entry $(date -u +%T)] $*"; }

( /start.sh > /tmp/start_sh.log 2>&1 & )
ev "entry container up pod=$ID name=${HM_POD_NAME:-} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

[ -z "${HM_KILL_AT:-}" ] && HM_KILL_AT=$(( T_ENTRY + 3600 ))
export HM_KILL_AT
[ -s "$PD/kill_at" ] || echo "$HM_KILL_AT" > "$PD/kill_at"
if [ -x "$HM/bin/hm_killswitch.sh" ]; then
  nohup "$HM/bin/hm_killswitch.sh" "$PD/kill_at" "${HM_HARD_CEIL:-0}" >> "$PD/killswitch.log" 2>&1 < /dev/null &
else
  ev "WARNING: $HM/bin/hm_killswitch.sh missing; no on-pod kill switch"
fi

if [ "${HM_AUTOSTART:-1}" != "1" ]; then
  ev "gateway mode (HM_AUTOSTART=${HM_AUTOSTART:-}): no worker"
else
  export HM_ROOT="$HM"
  # env (8 GB tar unpack, ~20 s) and libs (cached .debs, 10-60 s) in parallel, then code + links; `wait` names the PIDs
  # (a bare `wait` would block forever on the kill switch loop, also a child of this shell)
  bash "$HM/bin/pod_prepare.sh" env >> "$PD/prepare.log" 2>&1 & P_ENV=$!
  bash "$HM/bin/pod_prepare.sh" libs >> "$PD/prepare.log" 2>&1 & P_LIBS=$!
  bash "$HM/bin/pod_prepare.sh" code links >> "$PD/prepare.log" 2>&1
  wait $P_ENV; rc_env=$?
  wait $P_LIBS; rc_libs=$?
  ev "prepared env rc=$rc_env libs rc=$rc_libs (env/code/libs/links)"
  WS=${HM_WS:-/opt/hm/ws}
  export PATH="/opt/hm/env/bin:$PATH" PYTHONPATH="$WS/dancecap" DANCECAP_POD=1 PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1 \
         TORCH_HOME="$HM/weights/torch_home" HF_HOME="$HM/weights/hf" MPLBACKEND=Agg HM_ROOT="$HM"
  if [ -x /opt/hm/env/bin/python ] && [ -d "$WS/dancecap/dancecap/farm" ]; then
    cd "$WS/dancecap" || exit 1
    ev "worker start kinds=${HM_KINDS:-all} idle=${HM_IDLE_MIN:-5}"
    python -u -m dancecap.farm.worker --volume "$HM" ${HM_KINDS:+--kinds "$HM_KINDS"} \
        ${HM_MAX_JOBS:+--max-jobs "$HM_MAX_JOBS"} --idle-exit-min "${HM_IDLE_MIN:-5}" --pod-id "$ID" \
        >> "$PD/worker.log" 2>&1
    rc=$?
    ev "worker exit rc=$rc"
    echo "$rc" > "$PD/worker_exit"
    grace=$(( $(date +%s) + ${HM_POST_GRACE_MIN:-15} * 60 ))
    cur=$(cat "$PD/kill_at" 2>/dev/null | tr -dc '0-9'); cur=${cur:-$HM_KILL_AT}
    if [ "$grace" -lt "$cur" ]; then echo "$grace" > "$PD/kill_at"; ev "deadline pulled in to $grace (post-worker grace)"; fi
  else
    ev "NO WORKER: env=$([ -x /opt/hm/env/bin/python ] && echo yes || echo no) code=$([ -d "$WS/dancecap/dancecap/farm" ] && echo yes || echo no) (see prepare.log)"
  fi
fi
# keep the container alive for ssh / pulls until the kill switch or the watcher deletes the pod
while true; do sleep 60; done
