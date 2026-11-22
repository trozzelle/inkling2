#!/usr/bin/env bash
# scripts/studio/tf_prefill_gate.sh RUNTIME_ID: runs inside window.sh. The Phase 3 prompt-path gate on the fork commit's build
# (tf_sync.sh FORK_DIR first: a dirty tree is refused; the runtime of that commit installed: RUNTIME_ID names it). In order:
#   0. preflight: every input present (the prefill prompts from `ink2q.tfchat documents` among them) and every output directory
#      created before any model load;
#   1. tf-inkling-kernels on the Studio build;
#   2. exact on 48 layers (differ 0);
#   3. partition on the full model with --drafts over the 8k prompt: cuts of 4,095+1+4,096, 37+4,096, tails of 1, 17 and 31 rows,
#      a cut inside a 16-key tile (2,053) and one inside a 64-key block (4,000), consecutive tiny chunks over the first 100 rows;
#      and 4,095+1+4,096 on the two chat-form 8k prompts;
#   4. identity of the serving paths: greedy with drafts equals plain on the bench prompt and one chat prompt; served seeded
#      temperature-0.7 replies with drafts equal the --no-drafts replies of the same build; the three 8k prompts served once,
#      their prompt_tokens checked against the ids (rendered length = ids length - 1: the opener is not rendered);
#   5. fidelity, chunk mode (the default; the layer gate also at --chunk 128, four chunks) and --window mode: the layer gate over
#      48 layers on one held-out sequence, and TensorFold's finals over the full held-out set scored against the teacher with the
#      paired comparison against ink2q's kortexa result (bound: 95% upper limit of TensorFold - ink2q KL <= +0.002);
#   6. prefill --spans --drafts on the 8k prompt at chunks of 4,096 and 2,048 (direct start, for the breakdown); their last-row
#      digests must agree.
# Steps 1 to 4 stop the job on the first failure (an identity failure). Step 5 and 6's legs are measurements and gates: a failed
# leg is recorded, the rest still run, and the job exits 1 at the end if any failed. The previous window's plain tokens and
# seeded replies are compared informationally (the prompt path changed, so they are not required to match).
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV STUDIO_HOME || exit 1
RID=${1:?runtime id}
R=$INK2Q_ROOT
D=$R/data
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
KERNELS=$TF_FORK/zig-out/bin/tf-inkling-kernels
PROMPTS=$INK2Q_WORK/tf-m2p0
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
P3=$INK2Q_WORK/tf-m2p3
PMSG=$P3/prefill-messages.json
PREPORT=$P3/prefill-report.json
PIDS=$P3/prefill-ids
L=$R/reports/tf-m1/layers
FIDS=$INK2Q_WORK/tf-m1/full/ids
REF=$R/reports/exl3q/kortexa-heldout.json
PREV=$R/reports/tf-m2p2/w1
W=$R/reports/tf-m2p3/w2
OUT=$P3/w2
PORT=8199
# the legs run with the settings a server starts with
unset INKLING_PREFILL_ROWS INKLING_DRAFT_DEPTH INKLING_DRAFT_CONFIDENCE INKLING_DRAFT_BITS INKLING_ROUND_TIMING

# 0. preflight
mkdir -p $W $OUT
for f in "$UV" "$TF" "$KERNELS" "$TF_FORK/REVISION" "$PACK/config.json" "$PACK/mtp.safetensors" "$LONG" "$PROMPTS/ids/index.txt" \
         "$PMSG" "$PREPORT" "$PIDS/index.txt" "$L/ids.json" "$L/ref.npy" "$L/floor.npy" "$FIDS/index.txt" "$D/heldout.safetensors" \
         "$D/heldout_manifest.json" "$R/teacher/heldout/done.json" "$REF" "$R/source"; do
  [ -e "$f" ] || { echo "preflight: missing $f"; exit 1; }
done
bench=$(sed -n 1p $PROMPTS/ids/index.txt); chat=$(sed -n 2p $PROMPTS/ids/index.txt)
for f in "$bench" "$chat" "$PIDS/seq-00000.json" "$PIDS/seq-00001.json" "$PIDS/seq-00002.json"; do
  [ -e "$f" ] || { echo "preflight: missing ids file $f"; exit 1; }
done
ids_len() {  # ids_len FILE: the number of token ids in a JSON list of ints (nothing printed when the file is not one)
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list) and all(isinstance(x, int) for x in d)
print(len(d))" "$1" 2>/dev/null || true
}
[ "$(ids_len "$LONG")" = 8192 ] || { echo "preflight: $LONG is not 8,192 token ids"; exit 1; }
for i in 0 1 2; do
  [ "$(ids_len "$PIDS/seq-0000$i.json")" = 8192 ] || { echo "preflight: $PIDS/seq-0000$i.json is not 8,192 token ids"; exit 1; }
done
[ "$(ids_len "$bench")" = 116 ] || { echo "preflight: $bench is not the 116-token bench prompt"; exit 1; }
[ "$(ids_len "$chat")" = 934 ] || { echo "preflight: $chat is not the 934-token chat prompt"; exit 1; }
python3 - "$PMSG" "$PREPORT" <<'PY' || { echo "preflight: $PMSG is not the three 8k prompts the report describes"; exit 1; }
import json, sys
m, r = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
ok = len(m) == 3 and len(r) == 3 and all(
    e["ids_length"] == 8192 and e["rendered_length"] == 8191 and x["exact"] and x["messages_roundtrip"] and x["ids_length"] == 8192
    for e, x in zip(m, r))
sys.exit(0 if ok else 1)
PY
usage=$($TF 2>&1 || true)
for want in "partition <ids.json>" "prefill <ids.json>"; do
  case "$usage" in *"$want"*) ;; *) echo "preflight: $TF has no '$want' mode; build the fork commit with tf_sync.sh"; exit 1 ;; esac
done
grep -q 'layers <ids.json>.*--chunk' <<< "$usage" || { echo "preflight: $TF's layers mode has no --chunk"; exit 1; }
grep -q 'finals <index.txt>.*--chunk' <<< "$usage" || { echo "preflight: $TF's finals mode has no --chunk"; exit 1; }
BIN=$(find $STUDIO_HOME/runtimes/tensorfold -maxdepth 7 -type f -name tensorfold-native -path "*$RID*" | head -1)
[ -x "$BIN" ] || { echo "preflight: no tensorfold-native server for runtime $RID"; exit 1; }
read -r REV DIRTY < $TF_FORK/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "preflight: runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac
cp $TF_FORK/REVISION $W/tf-revision.txt
echo "$RID" > $W/runtime-id.txt
cp $PREPORT $W/prefill-prompts-report.json
cd "$STUDIO_REPO"
echo "preflight: pass"

FAILED=""
run() {  # run NAME CMD...: CMD's output to the log NAME.log (and here); its status is returned, never fatal here
  local name=$1 rc
  shift
  echo "$(date '+%F %T') == $name: $*"
  set +e
  "$@" 2>&1 | tee $W/$name.log
  rc=${PIPESTATUS[0]}
  set -e
  return $rc
}
must() {  # must NAME CMD...: run, and stop the job when it fails (an identity check)
  run "$@" || { echo "$(date '+%F %T') $1 FAILED: stopping"; exit 1; }
}
leg() {  # leg NAME CMD...: run; a failure is recorded and the job goes on (a gate or a measurement)
  run "$@" || { echo "$(date '+%F %T') leg $1 FAILED"; FAILED="$FAILED $1"; return 1; }
}

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

# 1. the kernel gate, on the Studio build
must kernels bash -c "cd $TF_FORK && $KERNELS"
grep -q "all cases pass" $W/kernels.log || { echo "kernels: no 'all cases pass' line"; exit 1; }

# 2. exact on 48 layers
must exact $TF $PACK exact $L/ids.json 448
grep -q '"differ": 0,' $W/exact.log || { echo "exact: differ is not 0"; exit 1; }

# 3. partition on the full model, with the draft head: the reference leg is chunks of the prefill rows
part() {  # part NAME IDS CUT...
  local name=$1 ids=$2; shift 2
  must partition-$name $TF $PACK partition $ids "$@" --drafts
}
part c4095-1 $LONG 4095 1 4096
part c37 $LONG 37 4096
part tail1 $LONG 4096 4095
part tail17 $LONG 4096 4079
part tail31 $LONG 4096 4065
part tile $LONG 2053 4096             # a cut inside a 16-key tile
part block $LONG 4000 4096            # a cut inside a 64-key block
part tiny100 $LONG 5 7 3 11 13 17 19 25
part chat-form-1 $PIDS/seq-00001.json 4095 1 4096
part chat-form-2 $PIDS/seq-00002.json 4095 1 4096
echo "partition: pass"

# 4. identity of the serving paths
$TF $PACK greedy $bench 256 2>&1 | tee $W/greedy-plain-1.log | tail -1
$TF $PACK greedy $bench 256 --drafts 2>&1 | tee $W/greedy-drafts-1.log | tail -1
same_tokens $W/greedy-plain-1.log $W/greedy-drafts-1.log || { echo "greedy tokens differ with drafts (bench)"; exit 1; }
$TF $PACK greedy $chat 256 2>&1 | tee $W/greedy-plain-2.log | tail -1
$TF $PACK greedy $chat 256 --drafts 2>&1 | tee $W/greedy-drafts-2.log | tail -1
same_tokens $W/greedy-plain-2.log $W/greedy-drafts-2.log || { echo "greedy tokens differ with drafts (chat)"; exit 1; }
echo "greedy: drafted equals plain"
same_tokens $PREV/greedy-plain-1.log $W/greedy-plain-1.log || echo "info: plain greedy tokens differ from the previous window's (the prompt path changed)"

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
seeded() {  # seeded OUT: the previous windows' four seeded temperature-0.7 requests
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
SERVER=
trap '[ -n "$SERVER" ] && { kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; }' EXIT
startup_info() {  # startup_info LOG: the cache room, scratch and the draft head's lines (informational)
  grep -h "GiB for caches\|prompt chunks of\|draft head\|window forwards" "$1" 2>/dev/null | head -5 || true
}

serve_up $W/serve-plain.log --no-drafts
startup_info $W/serve-plain.log
seeded $W/seeded-plain.jsonl
stop
serve_up $W/serve-drafts.log
startup_info $W/serve-drafts.log
seeded $W/seeded-drafts.jsonl
cmp $W/seeded-plain.jsonl $W/seeded-drafts.jsonl || { echo "seeded sampled replies differ from --no-drafts"; exit 1; }
echo "seeded: drafted equals plain"
cmp -s $PREV/seeded-plain.jsonl $W/seeded-plain.jsonl || echo "info: seeded plain replies differ from the previous window's (the prompt path changed)"
# the three 8k prompts served once: prompt_tokens reproduce the ids (the opener is the one token the template does not render)
$UV run python -m ink2q.tfgate bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --messages $PMSG \
  --requests 1 --max-tokens 8 --out $W/prefill-served.json | tail -3 || true
$UV run python - $W/prefill-served.json <<'PY' || { echo "served prompt_tokens do not reproduce the ids"; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
print(json.dumps({"served": [p["prompt_tokens"] for p in d["per_prompt"]], "unexplained": d["prompt_tokens_check"]["unexplained"]}))
ok = d["requests"] == 3 and not d["prompt_tokens_check"]["unexplained"] \
    and all(p["prompt_tokens"] == [8191] for p in d["per_prompt"])
sys.exit(0 if ok else 1)
PY
echo "served: prompt_tokens 8191 = ids 8192 - 1 on all three"
stop

# 5. fidelity, chunk mode and window mode (flags: none is chunk mode at the prefill rows; --chunk N; --window)
fidelity() {  # fidelity NAME [TF flags...]
  local name=$1; shift
  if leg layers-$name $TF $PACK layers $L/ids.json $OUT/layers-$name.bin 48 "$@"; then
    leg layer-gate-$name $UV run python -m ink2q.tflayers gate --ref $L --tf $OUT/layers-$name.bin --out $W/layer-gate-$name.json || true
  fi
}
finals() {  # finals NAME [TF flags...]
  local name=$1 f=$OUT/full-$1; shift
  mkdir -p $f/tf-bin
  leg finals-$name $TF $PACK finals $FIDS/index.txt $f/tf-bin "$@" || return 0
  leg finals-import-$name $UV run python -m ink2q.tfio finals --src $f/tf-bin --out $f/tf || return 0
  leg evaluate-$name $UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors \
    --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout --variant $f/tf --out $W/full-tf-$name.json \
    --positions $W/full-tf-$name.npz || return 0
  leg compare-$name $UV run python -m ink2q.compare $REF $W/full-tf-$name.json --out $W/full-compare-tf-$name.json || return 0
  if leg verdict-$name $UV run python -m ink2q.tfio verdict --compare $W/full-compare-tf-$name.json --max-hi 0.002; then :; fi
  tail -1 $W/verdict-$name.log > $W/full-verdict-tf-$name.json || true
}
fidelity chunk
fidelity chunk128 --chunk 128
fidelity window --window
finals chunk
finals window --window
# the floor's own verdict from Phase 1, for reference (informational)
$UV run python -m ink2q.tfio verdict --compare $R/reports/tf-m1/full-compare-floor.json --max-hi 0.002 || true

# 6. prefill with the draft head on the 8k prompt, at 4,096 and 2,048 rows (direct start)
for c in 4096 2048; do
  leg prefill-drafts-c$c $TF $PACK prefill $LONG --spans --drafts --chunk $c || true
done
if [ -s $W/prefill-drafts-c4096.log ] && [ -s $W/prefill-drafts-c2048.log ]; then
  python3 - $W/prefill-drafts-c4096.log $W/prefill-drafts-c2048.log <<'PY' | tee $W/digests.json || { echo "digests: the two chunk sizes end in different h_digests"; FAILED="$FAILED digests"; }
import json, sys
legs = {}
for path in sys.argv[1:]:
    rows = [json.loads(l) for l in open(path, errors="replace") if l.startswith('{"prompt_tokens"')]
    legs[path.rsplit("/", 1)[-1][:-4]] = rows[-1]["h_digest"] if rows else None
equal = None not in legs.values() and len(set(legs.values())) == 1
print(json.dumps({"legs": legs, "equal": equal}))
sys.exit(0 if equal else 1)
PY
else
  echo "digests: a prefill leg failed"; FAILED="$FAILED digests"
fi

if [ -n "$FAILED" ]; then
  echo "failed:$FAILED"
  exit 1
fi
echo done
