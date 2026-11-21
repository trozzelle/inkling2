#!/usr/bin/env bash
# scripts/studio/tf_p1_gate.sh: runs inside window.sh. The decode path on the full model: windows against one-row
# steps (stop on any difference), greedy decode speed after the bench prompt and after an 8,192-token prompt, and the
# per-class profile of 48-layer decode steps at 1 and 6 rows.
set -euo pipefail
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
UV=$UV
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
W=$R/reports/tf-m2p1
mkdir -p $W
cd "$REPO"
cp $INK2Q_WORK/tf-fork/REVISION $W/tf-revision.txt
$TF $PACK exact $R/reports/tf-m1/layers/ids.json 448 2>&1 | tee $W/exact.log
BENCH=$(head -1 $INK2Q_WORK/tf-m2p0/ids/index.txt)
$TF $PACK greedy $BENCH 256 2>&1 | tee $W/greedy-bench.log
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
if [ ! -f $LONG ]; then
  $UV run python -c "
import json
from pathlib import Path
from ink2q.tokens import load_tokens
ids = [int(t) for s in load_tokens(Path('$R/data/heldout.safetensors')) for t in s][:8192]
assert len(ids) == 8192
json.dump(ids, open('$LONG', 'w'))"
fi
$TF $PACK greedy $LONG 256 2>&1 | tee $W/greedy-8k.log
for args in "512 1" "8192 1" "512 6"; do
  set -- $args
  $TF $PACK profile 48 $1 $2 32 2>&1 | grep '^{' | sed "s/^/{\"pos\": $1, \"rows_arg\": $2, \"line\": /; s/$/}/" | tee -a $W/profile-48.jsonl
done
echo done
