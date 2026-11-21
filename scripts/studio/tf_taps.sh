#!/usr/bin/env bash
# scripts/studio/tf_taps.sh: runs inside window.sh. The full-model forward with taps on the benchmark prompt set: the first
# prompt, its tap-48 alignment check (stop on failure), then the rest; 256 greedy tokens each.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
OUT=$INK2Q_WORK/tf-m2p0
W=$INK2Q_ROOT/reports/tf-m2p0
mkdir -p $OUT/taps $W
cd "$STUDIO_REPO"
cp $TF_FORK/REVISION $W/tf-revision.txt
head -1 $OUT/ids/index.txt > $OUT/ids/first.txt
tail -n +2 $OUT/ids/index.txt > $OUT/ids/rest.txt
$TF $PACK taps $OUT/ids/first.txt $OUT/taps 256 2>&1 | tee $W/taps.log
$UV run python -m ink2q.tftaps check --dir $OUT/taps --seq 0 --source $INK2Q_ROOT/source | tee $W/taps-check.json
# the rest keep their sequence numbers: write them to a second directory, then rename into place
mkdir -p $OUT/taps-rest
$TF $PACK taps $OUT/ids/rest.txt $OUT/taps-rest 256 2>&1 | tee -a $W/taps.log
for f in $OUT/taps-rest/seq-*; do
  b=$(basename $f); n=$((10#${b:4:5} + 1)); mv "$f" "$OUT/taps/seq-$(printf %05d $n).${b#*.}"
done
rmdir $OUT/taps-rest
ls $OUT/taps | wc -l
echo done
