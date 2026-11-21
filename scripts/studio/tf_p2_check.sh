#!/usr/bin/env bash
# scripts/studio/tf_p2_check.sh RUNTIME_ID: runs inside window.sh. The drafter on the full model:
#   1. windows against one-row steps (stop on any difference);
#   2. serving taps against the study taps of the same build (every row) and against the earlier recorded taps
#      (prompt rows only, which take the unchanged prompt path);
#   3. greedy tokens with drafts equal to plain greedy tokens, on the bench prompt and three chat prompts;
#   4. served: seeded sampled replies with drafts equal to --no-drafts replies; one round-timing leg with drafts;
#   5. served decode rates over the gate prompts: plain (--no-drafts) and drafts at depths 3 to 7.
# Identity legs (1-4) stop the job on a difference; measurement legs record and continue.
set -euo pipefail
RID=${1:?runtime id}
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
UV=$UV
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
P0=$INK2Q_WORK/tf-m2p0
OUT=$INK2Q_WORK/tf-m2p2/w1
W=$R/reports/tf-m2p2/w1
MSG=$INK2Q_WORK/tf-m2p2/gate-messages.json
PORT=8199
mkdir -p $OUT $W
cd "$REPO"
cp $INK2Q_WORK/tf-fork/REVISION $W/tf-revision.txt
read -r REV DIRTY < $INK2Q_WORK/tf-fork/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac

# 1. exactness
$TF $PACK exact $R/reports/tf-m1/layers/ids.json 448 2>&1 | tee $W/exact.log
grep -q '"differ": 0' $W/exact.log || { echo "exact: rows differ"; exit 1; }

# 2. taps: study mode of this build for seq 0, then serving taps over the same tokens
head -1 $P0/ids/index.txt > $OUT/index0.txt
$TF $PACK taps $OUT/index0.txt $OUT/study 256 2>&1 | tail -2
$UV run python - "$OUT/study/seq-00000.tokens.json" "$OUT/ftaps-ids.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1])); json.dump([int(x) for x in t["tokens"]], open(sys.argv[2], "w")); print("prompt", t["prompt"], "tokens", len(t["tokens"]))
PY
L=$($UV run python -c "import json; print(json.load(open('$OUT/study/seq-00000.tokens.json'))['prompt'])")
$TF $PACK ftaps $OUT/ftaps-ids.json $OUT/ftaps.bin $L 1 2>&1 | tail -1
$UV run python - "$OUT/ftaps.bin" "$OUT/study/seq-00000.taps.bin" "$P0/taps/seq-00000.taps.bin" "$L" <<'PY' | tee $W/taps-check.json
import json, sys, numpy as np
H = 4608
f, s, p0 = (np.fromfile(x, "<u2").reshape(-1, 8, H) for x in sys.argv[1:4]); L = int(sys.argv[4])
n = min(len(f), len(s))
same = int((f[:n] == s[:n]).all(axis=(1, 2)).sum())
m = min(L, len(p0))
prompt_same = int((f[:m] == p0[:m]).all(axis=(1, 2)).sum())
r = {"rows": n, "rows_equal_study": same, "prompt_rows": m, "prompt_rows_equal_recorded": prompt_same}
print(json.dumps(r))
# only the same-build identity is fatal; the earlier recorded taps' prompt rows are recorded for the report
sys.exit(0 if n > 0 and same == n else 1)
PY

# 3. greedy with drafts equals plain greedy
for i in 1 2 3 4; do
  ids=$(sed -n "${i}p" $P0/ids/index.txt)
  $TF $PACK greedy $ids 256 2>&1 | tee $W/greedy-plain-$i.log | tail -1
  INKLING_DRAFT_DEPTH=5 $TF $PACK greedy $ids 256 --drafts 2>&1 | tee $W/greedy-drafts-$i.log | tail -2
  $UV run python - "$W/greedy-plain-$i.log" "$W/greedy-drafts-$i.log" <<'PY' || { echo "greedy tokens differ with drafts (prompt $i)"; exit 1; }
import json, sys
def tokens(path):  # the first line that is a JSON list of ints (log lines may come before it)
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
print(json.dumps({"plain": len(a), "drafts": len(b), "compared": n, "equal": a[:n] == b[:n]}))
sys.exit(0 if n >= 250 and a[:n] == b[:n] else 1)
PY
done
echo "greedy: drafted equals plain"

# 4. served
serve_up() {  # serve_up LOG [extra serve flags...]: start the runtime's server on $PORT and wait until it answers
  local log=$1; shift
  local bin
  bin=$(find $STUDIO_HOME/runtimes/tensorfold -maxdepth 7 -type f -name tensorfold-native -path "*$RID*" | head -1)
  [ -x "$bin" ] || { echo "no tensorfold-native server for runtime $RID" >&2; exit 1; }
  "$bin" serve "$PACK" --name inkling2-exl3-tf --context 32768 --parallel 1 --load-limit-gb 236 --port $PORT "$@" > "$log" 2>&1 &
  SERVER=$!
  for i in $(seq 1 120); do
    curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && return 0
    kill -0 $SERVER 2>/dev/null || { echo "server exited"; tail -40 "$log"; exit 1; }
    sleep 10
  done
  echo "server did not answer"; exit 1
}
stop() { kill $SERVER; wait $SERVER 2>/dev/null || true; SERVER=; }
seeded() {  # seeded OUT: four seeded temperature-0.7 replies (text and token count)
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
gate() {  # gate OUT [extra tfgate args]
  local out=$1; shift
  $UV run python -m ink2q.tfgate bench --base-url http://127.0.0.1:$PORT/v1 --model inkling2-exl3-tf --messages $MSG \
    --requests 1 --max-tokens 256 --out "$out" "$@" | tail -3 || true
}
SERVER=
trap '[ -n "$SERVER" ] && { kill $SERVER 2>/dev/null; wait $SERVER 2>/dev/null; }' EXIT

serve_up $W/serve-plain.log --no-drafts
seeded $W/seeded-plain.jsonl
gate $W/gate-plain.json
stop
export INKLING_DRAFT_DEPTH=5
serve_up $W/serve-d5.log
seeded $W/seeded-d5.jsonl
cmp $W/seeded-plain.jsonl $W/seeded-d5.jsonl || { echo "seeded sampled replies differ with drafts"; exit 1; }
echo "seeded: drafted equals plain"
gate $W/gate-d5.json
stop

# 4b. one round-timing leg with drafts at depth 5 (separate from the rates)
INKLING_ROUND_TIMING=1 serve_up $W/serve-timing.log
gate $W/gate-timing.json
stop

# 5. depth sweep over the gate prompts
for d in 3 4 6 7; do
  INKLING_DRAFT_DEPTH=$d serve_up $W/serve-d$d.log
  gate $W/gate-d$d.json
  stop
done
unset INKLING_DRAFT_DEPTH
echo done
