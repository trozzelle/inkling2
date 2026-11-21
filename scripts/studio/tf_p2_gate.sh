#!/usr/bin/env bash
# scripts/studio/tf_p2_gate.sh RUNTIME_ID: runs inside window.sh. The drafted decode gate with the drafter's defaults:
#   0. preflight: every input present and every output directory created before any model load;
#   1. identity: plain greedy tokens on the bench prompt equal the previous window's; greedy with drafts equals plain on
#      the bench prompt and one chat prompt; served seeded temperature-0.7 replies with drafts equal the previous
#      window's --no-drafts replies (the target path is unchanged; only the drafter changed);
#   2. the gate: served pooled decode over the 13 gate prompts, 2 requests each, temperature 0 (>= 55 tok/s);
#   3. measurements: sampled (temperature 0.7) with drafts, and bf16 draft layers at the same depth for comparison.
set -euo pipefail
RID=${1:?runtime id}
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
UV=$UV
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
P0=$INK2Q_WORK/tf-m2p0
W1=$R/reports/tf-m2p2/w1
W=$R/reports/tf-m2p2/w2
MSG=$INK2Q_WORK/tf-m2p2/gate-messages.json
PORT=8199

# 0. preflight
mkdir -p $W
for f in "$UV" "$TF" "$MSG" "$P0/ids/index.txt" "$PACK/mtp.safetensors" "$W1/greedy-plain-1.log" "$W1/seeded-plain.jsonl" \
         "$W1/gate-plain.json"; do
  [ -e "$f" ] || { echo "preflight: missing $f"; exit 1; }
done
for i in 1 2; do
  ids=$(sed -n "${i}p" $P0/ids/index.txt); [ -e "$ids" ] || { echo "preflight: missing ids file $ids"; exit 1; }
done
BIN=$(find $STUDIO_HOME/runtimes/tensorfold -maxdepth 7 -type f -name tensorfold-native -path "*$RID*" | head -1)
[ -x "$BIN" ] || { echo "preflight: no tensorfold-native server for runtime $RID"; exit 1; }
read -r REV DIRTY < $INK2Q_WORK/tf-fork/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "preflight: runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac
cp $INK2Q_WORK/tf-fork/REVISION $W/tf-revision.txt
cd "$REPO"
echo "preflight: pass"

same_tokens() {  # same_tokens A B: the first JSON int-list line of each log, equal on a common prefix of >= 250
  $UV run python - "$1" "$2" <<'PY'
import json, sys
def tokens(path):
    for line in open(path):
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if isinstance(v, list) and all(isinstance(x, int) for x in v):
            return v
    raise SystemExit(f"no token list in {path}")
a, b = tokens(sys.argv[1]), tokens(sys.argv[2])
n = min(len(a), len(b))
print(json.dumps({"a": len(a), "b": len(b), "compared": n, "equal": a[:n] == b[:n]}))
sys.exit(0 if n >= 250 and a[:n] == b[:n] else 1)
PY
}

# 1. identity
bench=$(sed -n 1p $P0/ids/index.txt); chat=$(sed -n 2p $P0/ids/index.txt)
$TF $PACK greedy $bench 256 2>&1 | tee $W/greedy-plain-1.log | tail -1
same_tokens $W1/greedy-plain-1.log $W/greedy-plain-1.log || { echo "plain greedy tokens changed since the previous window"; exit 1; }
$TF $PACK greedy $bench 256 --drafts 2>&1 | tee $W/greedy-drafts-1.log | tail -1
same_tokens $W/greedy-plain-1.log $W/greedy-drafts-1.log || { echo "greedy tokens differ with drafts (bench)"; exit 1; }
$TF $PACK greedy $chat 256 2>&1 | tee $W/greedy-plain-2.log | tail -1
$TF $PACK greedy $chat 256 --drafts 2>&1 | tee $W/greedy-drafts-2.log | tail -1
same_tokens $W/greedy-plain-2.log $W/greedy-drafts-2.log || { echo "greedy tokens differ with drafts (chat)"; exit 1; }
echo "greedy: drafted equals plain"

serve_up() {  # serve_up LOG [extra serve flags...]
  local log=$1; shift
  "$BIN" serve "$PACK" --name inkling2-exl3-tf --context 32768 --parallel 1 --load-limit-gb 236 --port $PORT "$@" > "$log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -40 "$log"; exit 1; }
    sleep 10
  done
  echo "server did not answer"; exit 1
}
stop() { kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=; }
seeded() {  # seeded OUT: the previous window's four seeded temperature-0.7 requests
  python3 - "$PORT" "$1" <<'PY'
import json, sys, urllib.request
port, out = sys.argv[1], sys.argv[2]
with open(out, "w") as f:
    for seed, (p, knobs) in enumerate([("Write a short poem about the sea.", {"top_p": 0.95}),
                                       ("List five uses for a paperclip.", {"top_p": 0.95, "top_k": 20}),
                                       ("Explain recursion to a child.", {"min_p": 0.05}),
                                       ("Describe how a hash map handles collisions.", {})]):
        body = dict(model="inkling2-exl3-tf", max_tokens=192, temperature=0.7, seed=2000 + seed,
                    messages=[{"role": "user", "content": p}], **knobs)
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(),
                                     {"content-type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        m = d["choices"][0]["message"]
        f.write(json.dumps({"seed": 2000 + seed, "content": m.get("content"), "reasoning": m.get("reasoning_content"),
                            "completion_tokens": (d.get("usage") or {}).get("completion_tokens")}, sort_keys=True) + "\n")
PY
}
gate() {  # gate OUT REQUESTS [extra tfgate args]
  local out=$1 n=$2; shift 2
  $UV run python -m ink2q.tfgate bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --messages $MSG \
    --requests $n --max-tokens 256 --out "$out" "$@" | tail -3 || true
}
SERVER=
trap '[ -n "$SERVER" ] && { kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; }' EXIT

# drafts with the drafter's defaults (no INKLING_DRAFT_* set)
serve_up $W/serve-default.log
grep -h "draft head\|window forwards" $W/serve-default.log 2>/dev/null | head -3 || true
seeded $W/seeded-drafts.jsonl
cmp $W1/seeded-plain.jsonl $W/seeded-drafts.jsonl || { echo "seeded sampled replies differ from --no-drafts"; exit 1; }
echo "seeded: drafted equals plain"

# 2. the gate
gate $W/gate.json 2
# 3. sampled with drafts
gate $W/gate-sampled.json 1 --temperature 0.7 --seed 7
stop

# 3b. bf16 draft layers at the same depth
INKLING_DRAFT_BITS=16 serve_up $W/serve-bf16.log
gate $W/gate-bf16.json 1
stop
echo done
