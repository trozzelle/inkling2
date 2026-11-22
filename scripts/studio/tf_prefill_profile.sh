#!/usr/bin/env bash
# scripts/studio/tf_prefill_profile.sh RUNTIME_ID: runs inside window.sh. What the prompt path costs by class on real prompts, from
# tf-inkling-run's `prefill` mode. Build the fork commit first (tf_sync.sh FORK_DIR tf-inkling-run): a dirty tree is refused.
#   0. preflight: every input present and every output directory created before any model load;
#   1. the 8k prompt at chunks of 512 rows (today's), a line a chunk for attention's growth with position; then greedy on it for one
#      step: its prompt_ms against prefill's, within 2%;
#   2. the 8k prompt at chunks of 128 rows, and at 512 with the draft head (the drafter's share: taps, the final stage on every row,
#      absorb); the bench prompt (116 tokens) and the chat prompt (934), each with and without drafts;
#   3. the 8k prompt at chunks of 2,048 and 4,096 rows, and at 4,096 with the draft head: per-class GPU time against chunk size, which
#      tests whether the per-row stages' cost per token is chunk-size independent.
# A forward's layers go to the GPU in command buffers of bounded work (about a second of GPU time today: a layer at a time from 2,048
# rows up), so none nears the 15.6 s at which one has been killed; each leg's `max_buffer_ms` is the longest it used. The long chunks
# still run last, so a failure there leaves the legs before it in. The 8k legs must all end in the same `h_digest` (the last row's
# hidden state): rows are independent, so no chunk size, command buffer split or draft head changes it.
# Every leg is a measurement: a failed leg is recorded and the rest still run, and the job exits 1 at the end if any failed. Each
# leg loads the model on its own. `prefill` runs a 16-row window first, as the server's warm-up does; greedy does not, so its
# prompt_ms includes the weights' first touch (about 1 s of 126 s on the 8k prompt in the decode window).
set -euo pipefail
. "$(dirname "$0")/env.sh" || exit 1
RID=${1:?runtime id}
R=$INK2Q_ROOT
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
PROMPTS=$INK2Q_WORK/tf-m2p0
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
W=$R/reports/tf-m2p3/w1
# the legs run with the settings a server starts with
unset INKLING_PREFILL_ROWS INKLING_DRAFT_DEPTH INKLING_DRAFT_CONFIDENCE INKLING_DRAFT_BITS INKLING_ROUND_TIMING INKLING_PROMPT_PATH

ids_len() {  # ids_len FILE: the number of token ids in a JSON list of ints (nothing printed when the file is not one)
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list) and all(isinstance(x, int) for x in d)
print(len(d))" "$1" 2>/dev/null || true
}

# 0. preflight
mkdir -p $W
for f in "$TF" "$TF_FORK/REVISION" "$PACK/config.json" "$PACK/mtp.safetensors" "$LONG" "$PROMPTS/ids/index.txt"; do
  [ -e "$f" ] || { echo "preflight: missing $f"; exit 1; }
done
bench=$(sed -n 1p $PROMPTS/ids/index.txt); chat=$(sed -n 2p $PROMPTS/ids/index.txt)
for f in "$bench" "$chat"; do
  [ -e "$f" ] || { echo "preflight: missing ids file $f"; exit 1; }
done
n=$(ids_len "$LONG"); [ -n "$n" ] && [ "$n" -ge 8192 ] || { echo "preflight: $LONG is not a JSON list of at least 8192 token ids"; exit 1; }
[ "$(ids_len "$bench")" = 116 ] || { echo "preflight: $bench is not the 116-token bench prompt"; exit 1; }
[ "$(ids_len "$chat")" = 934 ] || { echo "preflight: $chat is not the 934-token chat prompt"; exit 1; }
usage=$($TF 2>&1 || true)
case "$usage" in *"prefill <ids.json>"*) ;; *) echo "preflight: $TF has no prefill mode; build the fork commit with tf_sync.sh"; exit 1 ;; esac
read -r REV DIRTY < $TF_FORK/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "preflight: runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac
cp $TF_FORK/REVISION $W/tf-revision.txt
echo "$RID" > $W/runtime-id.txt
echo "preflight: pass"

FAILED=""
leg() {  # leg NAME CMD...: CMD's output to the log NAME.log (and here); a failure is recorded and the job goes on
  local name=$1 rc
  shift
  echo "$(date '+%F %T') == $name: $*"
  set +e
  "$@" 2>&1 | tee $W/$name.log
  rc=${PIPESTATUS[0]}
  set -e
  [ "$rc" = 0 ] || { echo "$(date '+%F %T') leg $name FAILED (rc=$rc)"; FAILED="$FAILED $name"; }
}

# 1. today's chunk on the 8k prompt, and greedy's prompt_ms against it
leg prefill-c512 $TF $PACK prefill $LONG --spans --chunk 512
leg greedy-8k $TF $PACK greedy $LONG 1
crosscheck() {
  python3 - $W/greedy-8k.log $W/prefill-c512.log <<'PY'
import json, sys
def last(path):  # the last result line: {"prompt_tokens": ...} (greedy's and prefill's both start so; a chunk's line starts "chunk")
    rows = [json.loads(l) for l in open(path, errors="replace") if l.startswith('{"prompt_tokens"')]
    if not rows:
        raise SystemExit(f"no result line in {path}")
    return rows[-1]
g, p = last(sys.argv[1]), last(sys.argv[2])
r = {"prompt_tokens": [g["prompt_tokens"], p["prompt_tokens"]], "greedy_prompt_ms": g["prompt_ms"], "prefill_prompt_ms": p["prompt_ms"]}
r["ratio"] = round(p["prompt_ms"] / g["prompt_ms"], 4)
r["within_2_percent"] = g["prompt_tokens"] == p["prompt_tokens"] and abs(r["ratio"] - 1) <= 0.02
print(json.dumps(r))
sys.exit(0 if r["within_2_percent"] else 1)
PY
}
if [ -s $W/greedy-8k.log ] && [ -s $W/prefill-c512.log ]; then
  crosscheck | tee $W/crosscheck.json || { echo "crosscheck: prompt_ms not confirmed within 2% (see crosscheck.json and the message above it)"; FAILED="$FAILED crosscheck"; }
else
  echo "crosscheck: a leg it reads failed"; FAILED="$FAILED crosscheck"
fi

# 2. small chunks, the drafter's share at today's chunk, the short prompts
leg prefill-c128 $TF $PACK prefill $LONG --spans --chunk 128
leg prefill-drafts-c512 $TF $PACK prefill $LONG --spans --drafts --chunk 512
leg prefill-bench $TF $PACK prefill $bench
leg prefill-bench-drafts $TF $PACK prefill $bench --drafts
leg prefill-chat $TF $PACK prefill $chat
leg prefill-chat-drafts $TF $PACK prefill $chat --drafts

# 3. the long chunks
for c in 2048 4096; do
  leg prefill-c$c $TF $PACK prefill $LONG --spans --chunk $c
done
leg prefill-drafts-c4096 $TF $PACK prefill $LONG --spans --drafts --chunk 4096

# the 8k legs' last-row digests agree
digests() {
  python3 - $W <<'PY'
import glob, json, os, re, sys
legs = {}
for path in sorted(glob.glob(os.path.join(sys.argv[1], "prefill-*.log"))):
    name = os.path.basename(path)[:-4]
    if not re.fullmatch(r"prefill-(drafts-)?c\d+", name):
        continue
    rows = [json.loads(l) for l in open(path, errors="replace") if l.startswith('{"prompt_tokens"')]
    if rows and "h_digest" in rows[-1]:
        legs[name] = rows[-1]["h_digest"]
equal = len(legs) >= 2 and len(set(legs.values())) == 1
print(json.dumps({"legs": legs, "equal": equal}))
sys.exit(0 if equal else 1)
PY
}
digests | tee $W/digests.json || { echo "digests: the 8k legs do not all end in one h_digest (see digests.json)"; FAILED="$FAILED digests"; }

# the legs' result lines in one file, with the unspanned remainder and the cost a token (informational)
python3 - $W <<'PY' || true
import glob, json, os, sys
w, out = sys.argv[1], []
for path in sorted(glob.glob(os.path.join(w, "prefill-*.log"))):
    rows = [json.loads(l) for l in open(path, errors="replace") if l.startswith('{"prompt_tokens"')]
    if not rows:
        continue
    r = rows[-1]
    r["leg"] = os.path.basename(path)[:-4]
    r["unspanned_ms"] = round(r["gpu_ms"] - sum(r["classes"].values()), 2)
    r["us_per_token"] = {k: round(v * 1e3 / r["prompt_tokens"], 2) for k, v in r["classes"].items()}
    out.append(r)
with open(os.path.join(w, "summary.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(f"summary: {len(out)} legs")
PY

if [ -n "$FAILED" ]; then
  echo "failed:$FAILED"
  exit 1
fi
echo done
