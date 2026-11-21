#!/usr/bin/env bash
# scripts/studio/tf_p1_serve.sh RUNTIME_ID [OUT_SUBDIR] [REQUESTS] [SAMPLED]: runs inside window.sh. First the build
# under test must reproduce the decode path's earlier outputs byte for byte (a 48-layer window-mode dump and greedy
# tokens after the bench prompt and after an 8,192-token prompt); any difference stops the job. Then the installed
# runtime serves the model on a loopback port and `studio bench` measures one-stream decode at <= 1k context and after
# an 8k-token prompt. With SAMPLED=1, streamed requests at temperature 0.7 measure sampled decode too.
set -euo pipefail
RID=${1:?runtime id}
SUB=${2:-serve}
NREQ=${3:-4}
SAMPLED=${4:-0}
R=$INK2Q_ROOT
STUDIO=$STUDIO_HOME/bin/studio
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
OUT=$INK2Q_WORK/tf-m2p1
W=$R/reports/tf-m2p1
S=$W/$SUB
LONG=$INK2Q_WORK/tf-m2p1-long8k
PORT=8199
mkdir -p $S
cp $INK2Q_WORK/tf-fork/REVISION $S/tf-revision.txt
# 0. identity with the gated build
# the runtime served below must be the build the identity leg checks: same commit, clean tree
read -r REV DIRTY < $INK2Q_WORK/tf-fork/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac
$TF $PACK layers $R/reports/tf-m1/layers/ids.json $OUT/layers-win-$SUB.bin 48 --window
cmp $OUT/layers-win.bin $OUT/layers-win-$SUB.bin
BENCH=$(head -1 $INK2Q_WORK/tf-m2p0/ids/index.txt)
$TF $PACK greedy $BENCH 256 2>&1 | tee $S/greedy-bench.log
[ "$(head -1 $S/greedy-bench.log)" = "$(head -1 $W/greedy-bench.log)" ] || { echo "greedy tokens differ (bench)"; exit 1; }
$TF $PACK greedy $LONG.json 256 2>&1 | tee $S/greedy-8k.log
[ "$(head -1 $S/greedy-8k.log)" = "$(head -1 $W/greedy-8k.log)" ] || { echo "greedy tokens differ (8k)"; exit 1; }
echo "identity: pass"
# 1. served decode
TFBIN=$(find $STUDIO_HOME/runtimes/tensorfold -maxdepth 7 -type f -name tensorfold-native -path "*$RID*" | head -1)
[ -x "$TFBIN" ] || { echo "no tensorfold-native server for runtime $RID" >&2; exit 1; }
"$TFBIN" serve "$PACK" --name inkling2-exl3-tf --context 32768 --parallel 1 --load-limit-gb 236 --port $PORT > $S/serve.log 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT
for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -40 $S/serve.log; exit 1; }
  sleep 10
done
$STUDIO bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --smoke --output $S/smoke.json
$STUDIO bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --concurrency 1 --requests $NREQ \
  --max-tokens 256 --output $S/bench-1k.json --label decode-path
$STUDIO bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --concurrency 1 --requests 2 \
  --max-tokens 256 --prompt-file $LONG.txt --output $S/bench-8k.json --label decode-path-8k
if [ "$SAMPLED" = 1 ]; then
  # streamed chat at temperature 0.7: decode rate from the first to the last streamed chunk
  python3 - "$PORT" "$S/sampled.jsonl" <<'PY'
import json, sys, time, urllib.request
port, out = sys.argv[1], sys.argv[2]
prompts = ["Explain why the sky is blue in three sentences.", "Write a Python function that merges two sorted lists.",
           "Summarise the plot of Hamlet in one paragraph.", "Describe how a hash map handles collisions."]
for i, p in enumerate(prompts + prompts[:2]):
    for knobs in ({"temperature": 0.7, "top_p": 0.95}, {"temperature": 0.7, "top_p": 0.95, "top_k": 20}):
        body = dict(model="inkling2-exl3-tf", stream=True, max_tokens=256, messages=[{"role": "user", "content": p}], **knobs)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        t0, stamps = time.time(), []
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if line.startswith(b"data: ") and line.strip() != b"data: [DONE]":
                    d = json.loads(line[6:])
                    if d.get("choices") and (d["choices"][0].get("delta") or {}):
                        stamps.append(time.time())
        rate = (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0
        rec = {"prompt": i, **knobs, "chunks": len(stamps), "ttft_s": round(stamps[0] - t0, 3), "decode_chunks_per_s": round(rate, 2)}
        print(json.dumps(rec)); open(out, "a").write(json.dumps(rec) + "\n")
PY
fi
echo done
