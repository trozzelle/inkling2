#!/usr/bin/env bash
# scripts/studio/window.sh OUTDIR RESTORE_MODEL -- CMD [ARGS...]
# Downtime window: freeze the router, stop its engines, run CMD, always restore. RESTORE_MODEL may be "none".
set -uo pipefail
[ $# -ge 4 ] && [ "$3" = "--" ] || { echo "usage: window.sh OUTDIR RESTORE_MODEL -- CMD..." >&2; exit 2; }
OUT=$1; RESTORE=$2; shift 3
mkdir -p "$OUT"
STUDIO=$STUDIO_HOME/bin/studio
log() { echo "$(date '+%F %T') $*"; }
ROUTER_PID=$(pgrep -f 'llama-swap -config $STUDIO_HOME/config/llama-swap.yaml' | head -1)
[ -n "$ROUTER_PID" ] || { log "no router; refusing (nothing to restore through)"; exit 1; }
descendants() { local p; for p in $(pgrep -P "$1"); do echo "$p"; descendants "$p"; done; }
restore() {
  local rc=$?
  kill -CONT "$ROUTER_PID" 2>/dev/null; sleep 5
  if [ "$RESTORE" != "none" ]; then
    "$STUDIO" smoke --model "$RESTORE" --smoke --timeout 1500 --output "$OUT/restore-smoke.json" \
      > "$OUT/restore-smoke.log" 2>&1
    log "restore: smoke rc=$?"
  fi
  vm_stat > "$OUT/vm_stat-post.txt"
  log "restore: done (cmd rc=$rc)"
}
CMD_PID=""
abort() {
  # bash defers trapped signals until a foreground child returns, so CMD runs in the background and is
  # signalled here; restore (EXIT trap) then resumes the router.
  if [ -n "$CMD_PID" ]; then
    kill -TERM $(descendants "$CMD_PID") "$CMD_PID" 2>/dev/null
    wait "$CMD_PID" 2>/dev/null
  fi
  exit 130
}
trap restore EXIT
trap abort INT TERM HUP
vm_stat > "$OUT/vm_stat-pre.txt"
kill -STOP "$ROUTER_PID"; log "router $ROUTER_PID frozen"
PIDS=$(descendants "$ROUTER_PID")
if [ -n "$PIDS" ]; then
  log "stopping engines: $PIDS"; kill -TERM $PIDS 2>/dev/null
  for _ in $(seq 120); do [ -z "$(descendants "$ROUTER_PID")" ] && break; sleep 1; done
  LEFT=$(descendants "$ROUTER_PID"); [ -z "$LEFT" ] || { kill -KILL $LEFT 2>/dev/null; sleep 5; }
  [ -z "$(descendants "$ROUTER_PID")" ] || { log "engines did not stop; aborting"; exit 1; }
fi
vm_stat > "$OUT/vm_stat-stopped.txt"
log "engines stopped; running: $*"
"$@" > "$OUT/cmd.out" 2> "$OUT/cmd.err" &
CMD_PID=$!
wait "$CMD_PID"
rc=$?
log "cmd rc=$rc"
exit $rc
