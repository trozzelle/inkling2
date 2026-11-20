#!/usr/bin/env bash
# scripts/studio/tf_model_full.sh: runs inside window.sh. The full held-out set two ways against ink2q's kortexa result:
# TensorFold's teacher-forced finals, and ink2q itself with fp32 static convolutions (a second correct implementation,
# the layer gate's floor) as the KL noise reference. Then a greedy decode for speed.
set -uo pipefail
R=$INK2Q_ROOT
D=$R/data
REPO=$STUDIO_REPO
UV=$UV
K=$INK2Q_WORK/kortexa
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
OUT=$INK2Q_WORK/tf-m1
W=$R/reports/tf-m1
REF=$R/reports/exl3q/kortexa-heldout.json
set -e
mkdir -p $OUT $W
cd "$REPO"
cp $INK2Q_WORK/tf-fork/REVISION $W/tf-revision-full.txt
# 1. TensorFold on the full held-out set
$UV run python -m ink2q.tfio tokens --tokens $D/heldout.safetensors --out $OUT/full/ids
mkdir -p $OUT/full/tf-bin
$TF $PACK finals $OUT/full/ids/index.txt $OUT/full/tf-bin
$UV run python -m ink2q.tfio finals --src $OUT/full/tf-bin --out $OUT/full/tf
# 2. ink2q with the kortexa experts and fp32 static convolutions
if [ ! -f $OUT/full/floor/done.json ]; then
  $UV run python -m ink2q.teacher --source $R/source --variant-kortexa $K --fp32-convs \
    --tokens $D/heldout.safetensors --out $OUT/full/floor
fi
# 3. both scored against the teacher and compared with ink2q's kortexa result
for v in tf floor; do
  $UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json \
    --teacher $R/teacher/heldout --variant $OUT/full/$v --out $W/full-$v.json --positions $W/full-$v.npz
  $UV run python -m ink2q.compare $REF $W/full-$v.json --out $W/full-compare-$v.json > /dev/null
done
set +e
$UV run python -m ink2q.tfio verdict --compare $W/full-compare-tf.json --max-hi 0.002 | tee $W/full-verdict-tf.json
$UV run python -m ink2q.tfio verdict --compare $W/full-compare-floor.json --max-hi 0.002 | tee $W/full-verdict-floor.json
set -e
# 4. decode: a held-out prompt of 512 tokens, then 128 greedy steps
$UV run python -m ink2q.tfio tokens --tokens $D/heldout.safetensors --out $OUT/sub512/ids --max-tokens 512 --limit 2
$TF $PACK greedy $OUT/sub512/ids/seq-00001.json 128 2>&1 | tee $W/greedy.txt
echo done
