#!/usr/bin/env bash
# scripts/studio/tf_serve_check.sh RUNTIME_ID: runs inside window.sh. The installed TensorFold runtime serves the
# Inkling 2 EXL3 directory on a loopback port; a smoke and a one-stream decode bench run against it.
set -euo pipefail
RID=${1:?runtime id}
STUDIO=$STUDIO_HOME/bin/studio
PACK=$INK2Q_WORK/tf-pack
W=$INK2Q_ROOT/reports/tf-m1
PORT=8199
TFBIN=$(find $STUDIO_HOME/runtimes/tensorfold -maxdepth 7 -type f -name tensorfold-native -path "*$RID*" | head -1)
[ -x "$TFBIN" ] || { echo "no tensorfold-native server for runtime $RID" >&2; exit 1; }
mkdir -p $W
"$TFBIN" serve "$PACK" --name inkling2-exl3-tf --context 32768 --parallel 1 --load-limit-gb 236 --port $PORT > $W/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -40 $W/serve.log; exit 1; }
  sleep 10
done
curl -sf http://127.0.0.1:$PORT/v1/models
$STUDIO bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --smoke --output $W/serve-smoke.json
$STUDIO bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --concurrency 1 --requests 4 \
  --max-tokens 256 --output $W/serve-bench.json
for q in "Explain why the sky is blue in three sentences." "Write a Python function that merges two sorted lists." \
         "Summarise the plot of Hamlet in one paragraph."; do
  curl -s http://127.0.0.1:$PORT/v1/chat/completions -H 'content-type: application/json' \
    -d "$(printf '{"model":"inkling2-exl3-tf","temperature":0,"max_tokens":200,"messages":[{"role":"user","content":"%s"}]}' "$q")" \
    | tee -a $W/serve-chat.jsonl
  echo >> $W/serve-chat.jsonl
done
echo done
