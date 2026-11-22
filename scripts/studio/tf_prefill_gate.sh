#!/usr/bin/env bash
# scripts/studio/tf_prefill_gate.sh RUNTIME_ID: runs inside window.sh. The Phase 3 prompt-path gate, slim window, on the fork
# commit's build (tf_sync.sh FORK_DIR first: a dirty tree is refused; RUNTIME_ID names the runtime of that commit). Legs:
#   0. preflight: every input present, every output directory created before any model load, the build's REVISION clean and the
#      runtime id that commit's, the binary's usage listing the modes used;
#   1. (fatal) exact on 48 layers (differ 0); the 48-layer layer gate in chunk mode (layers --chunk 4096, ink2q.tflayers gate:
#      every layer <= 1.5x the noise floor);
#   2. (fatal) partition --drafts on the raw 8k prompt, four cut sets: 16 (a 16-row first cut), 4095+1 (odd sizes, a 1-row tail),
#      2048+3001 (a ragged pair), 2053 (a cut inside a 16-key tile); and 4095+1 on one chat-form 8k prompt. Each must report
#      "identical";
#   3. (fatal) greedy 64 tokens after the 8k prefix: drafted equals plain;
#   4. (fatal) TensorFold's finals over the full held-out set in chunk mode (finals --chunk 4096) scored against the teacher, the
#      paired comparison against ink2q's kortexa result, verdict: 95% upper limit of TensorFold - ink2q KL <= +0.002;
#   5. (fatal) the 48-layer layer gate in window mode (layers --window);
#   6. (diagnostic, non-fatal) prefill --spans --drafts on the 8k prompt at chunk 4,096.
# A fatal leg's failure stops the job. A failed diagnostic leg is recorded and the job exits 1 at the end.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
RID=${1:?runtime id}
R=$INK2Q_ROOT
D=$R/data
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
P3=$INK2Q_WORK/tf-m2p3
CHAT8K=$P3/prefill-ids/seq-00001.json
L=$R/reports/tf-m1/layers
FIDS=$INK2Q_WORK/tf-m1/full/ids
REF=$R/reports/exl3q/kortexa-heldout.json
W=$R/reports/tf-m2p3/w2
OUT=$P3/w2
# the legs run with the settings a server starts with
unset INKLING_PREFILL_ROWS INKLING_DRAFT_DEPTH INKLING_DRAFT_CONFIDENCE INKLING_DRAFT_BITS INKLING_ROUND_TIMING INKLING_PROMPT_PATH

# 0. preflight
mkdir -p $W $OUT $OUT/full-chunk/tf-bin
for f in "$UV" "$TF" "$TF_FORK/REVISION" "$PACK/config.json" "$PACK/mtp.safetensors" "$LONG" "$CHAT8K" "$L/ids.json" "$L/ref.npy" \
         "$L/floor.npy" "$FIDS/index.txt" "$D/heldout.safetensors" "$D/heldout_manifest.json" "$R/teacher/heldout/done.json" \
         "$REF" "$R/source"; do
  [ -e "$f" ] || { echo "preflight: missing $f"; exit 1; }
done
ids_len() {  # ids_len FILE: the number of token ids in a JSON list of ints (nothing printed when the file is not one)
  python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list) and all(isinstance(x, int) for x in d)
print(len(d))" "$1" 2>/dev/null || true
}
for f in "$LONG" "$CHAT8K"; do
  [ "$(ids_len "$f")" = 8192 ] || { echo "preflight: $f is not 8,192 token ids"; exit 1; }
done
usage=$($TF 2>&1 || true)
for want in "partition <ids.json>" "prefill <ids.json>"; do
  case "$usage" in *"$want"*) ;; *) echo "preflight: $TF has no '$want' mode; build the fork commit with tf_sync.sh"; exit 1 ;; esac
done
grep -q 'layers <ids.json>.*--chunk' <<< "$usage" || { echo "preflight: $TF's layers mode has no --chunk"; exit 1; }
grep -q 'finals <index.txt>.*--chunk' <<< "$usage" || { echo "preflight: $TF's finals mode has no --chunk"; exit 1; }
read -r REV DIRTY < $TF_FORK/REVISION
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)"; exit 1; }
case "$RID" in *"${REV:0:12}"*) ;; *) echo "preflight: runtime $RID is not commit ${REV:0:12}"; exit 1 ;; esac
cp $TF_FORK/REVISION $W/tf-revision.txt
echo "$RID" > $W/runtime-id.txt
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
must() {  # must NAME CMD...: run, and stop the job when it fails
  run "$@" || { echo "$(date '+%F %T') $1 FAILED: stopping"; exit 1; }
}
leg() {  # leg NAME CMD...: run; a failure is recorded and the job goes on (a diagnostic)
  run "$@" || { echo "$(date '+%F %T') leg $1 FAILED"; FAILED="$FAILED $1"; return 1; }
}

same_tokens() {  # same_tokens A B MIN: the first JSON int-list line of each log, equal on a common prefix of >= MIN
  $UV run python - "$1" "$2" "$3" <<'PY'
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
a, b, need = tokens(sys.argv[1]), tokens(sys.argv[2]), int(sys.argv[3])
n = min(len(a), len(b))
print(json.dumps({"a": len(a), "b": len(b), "compared": n, "equal": a[:n] == b[:n]}))
sys.exit(0 if n >= need and a[:n] == b[:n] else 1)
PY
}

# 1. exact on 48 layers, then the layer gate in chunk mode
must exact $TF $PACK exact $L/ids.json 448
grep -q '"differ": 0,' $W/exact.log || { echo "exact: differ is not 0"; exit 1; }
must layers-chunk $TF $PACK layers $L/ids.json $OUT/layers-chunk.bin 48 --chunk 4096
must layer-gate-chunk $UV run python -m ink2q.tflayers gate --ref $L --tf $OUT/layers-chunk.bin --out $W/layer-gate-chunk.json

# 2. partition on the full model, with the draft head: the reference leg is chunks of the prefill rows
part() {  # part NAME IDS CUT...
  local name=$1 ids=$2; shift 2
  must partition-$name $TF $PACK partition $ids "$@" --drafts
  grep -q '"partition": "identical"' $W/partition-$name.log || { echo "partition-$name: not reported identical"; exit 1; }
}
part first16 $LONG 16
part c4095-1 $LONG 4095 1
part ragged $LONG 2048 3001
part tile $LONG 2053               # a cut inside a 16-key tile
part chat-form $CHAT8K 4095 1
echo "partition: pass"

# 3. greedy after the 8k prefix: drafted equals plain
must greedy-plain $TF $PACK greedy $LONG 64
must greedy-drafts $TF $PACK greedy $LONG 64 --drafts
same_tokens $W/greedy-plain.log $W/greedy-drafts.log 64 || { echo "greedy tokens differ with drafts (8k prefix)"; exit 1; }
echo "greedy: drafted equals plain"

# 4. held-out finals in chunk mode, scored against the teacher
f=$OUT/full-chunk
must finals-chunk $TF $PACK finals $FIDS/index.txt $f/tf-bin --chunk 4096
must finals-import-chunk $UV run python -m ink2q.tfio finals --src $f/tf-bin --out $f/tf
must evaluate-chunk $UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout --variant $f/tf --out $W/full-tf-chunk.json \
  --positions $W/full-tf-chunk.npz
must compare-chunk $UV run python -m ink2q.compare $REF $W/full-tf-chunk.json --out $W/full-compare-tf-chunk.json
must verdict-chunk $UV run python -m ink2q.tfio verdict --compare $W/full-compare-tf-chunk.json --max-hi 0.002
tail -1 $W/verdict-chunk.log > $W/full-verdict-tf-chunk.json

# 5. the layer gate in window mode
must layers-window $TF $PACK layers $L/ids.json $OUT/layers-window.bin 48 --window
must layer-gate-window $UV run python -m ink2q.tflayers gate --ref $L --tf $OUT/layers-window.bin --out $W/layer-gate-window.json

# 6. diagnostic: the prompt path's breakdown with the draft head on the 8k prompt
leg prefill-drafts-c4096 $TF $PACK prefill $LONG --spans --drafts --chunk 4096 || true

if [ -n "$FAILED" ]; then
  echo "failed:$FAILED"
  exit 1
fi
echo done
