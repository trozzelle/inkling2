#!/usr/bin/env bash
# scripts/studio/tf_router_repro.sh RUNTIME_ID: runs inside window.sh. Starts the serving profile's engine with the
# router's exact command line, output to a file, and polls /health and /v1/models every 10 s. If it is not healthy in
# 8 min, a stack sample is taken and two variants run (--no-drafts; without --dashboard) to isolate the cause.
set -uo pipefail
RID=${1:?runtime id}
W=$INK2Q_ROOT/reports/tf-m2p2/router
BIN=$STUDIO_HOME/runtimes/tensorfold/$RID/source/zig-out/native/bin/tensorfold-native
export TENSORFOLD_LIBTURBOJPEG=$STUDIO_HOME/runtimes/tensorfold/$RID/libjpeg-turbo/lib/libturbojpeg.0.dylib
PORT=19012
mkdir -p $W
[ -x "$BIN" ] || { echo "preflight: missing $BIN"; exit 1; }
[ -e "$TENSORFOLD_LIBTURBOJPEG" ] || { echo "preflight: missing $TENSORFOLD_LIBTURBOJPEG"; exit 1; }
echo "preflight: pass"

run() {  # run NAME FLAGS...: 0 when /health answers 200 within 8 min
  local name=$1; shift
  local log=$W/$name.log polls=$W/$name-polls.txt t0=$(date +%s) code
  "$BIN" serve $INK2Q_WORK/tf-pack --name inkling2-exl3-tf --host 127.0.0.1 --port $PORT --context 32768 \
    --max-tokens 8192 --parallel 1 "$@" --load-limit-gb 236 > $log 2>&1 &
  local pid=$!
  : > $polls
  for i in $(seq 1 48); do
    sleep 10
    kill -0 $pid 2>/dev/null || { echo "$name: exited after $(( $(date +%s) - t0 )) s"; tail -20 $log; return 1; }
    code=$(curl -s -m 5 -o $W/$name-health.body -w "%{http_code}" http://127.0.0.1:$PORT/health)
    echo "$(( $(date +%s) - t0 )) s health $code $(head -c 200 $W/$name-health.body 2>/dev/null) models $(curl -s -m 5 -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/v1/models)" >> $polls
    if [ "$code" = 200 ]; then
      echo "$name: healthy after $(( $(date +%s) - t0 )) s: $(head -c 200 $W/$name-health.body)"
      curl -s -m 600 http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
        -d '{"model":"inkling2-exl3-tf","max_tokens":64,"temperature":0,"messages":[{"role":"user","content":"Explain how a hash map handles collisions."}]}' \
        | head -c 300; echo
      grep -h "draft head\|tensorfold\] done" $log | tail -2 | cut -c1-300
      kill $pid; wait $pid 2>/dev/null
      return 0
    fi
  done
  echo "$name: not healthy after 8 min; last polls:"; tail -3 $polls
  sample $pid 3 -file $W/$name-sample.txt > /dev/null 2>&1 || true
  echo "$name: log tail:"; tail -15 $log
  kill $pid; sleep 5; kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  return 1
}

FLAGS=(--allowed-host inference-mac.local --allowed-host 192.0.2.10 --dashboard)
run exact "${FLAGS[@]}" && { echo done; exit 0; }
run nodrafts "${FLAGS[@]}" --no-drafts
run nodashboard --allowed-host inference-mac.local --allowed-host 192.0.2.10
echo done
