#!/usr/bin/env bash
# scripts/studio/tf_model_gate.sh: runs inside window.sh. TensorFold teacher-forced finals on the held-out set at 512
# tokens a sequence and in full, scored against the MXFP8 teacher and compared with ink2q's kortexa result; greedy decode.
set -euo pipefail
R=$INK2Q_ROOT
D=$R/data
REPO=$STUDIO_REPO
UV=$UV
PACK=$INK2Q_WORK/tf-pack
TF=$INK2Q_WORK/tf-fork/zig-out/bin/tf-inkling-run
OUT=$INK2Q_WORK/tf-m1
W=$R/reports/tf-m1
mkdir -p $OUT $W
cd "$REPO"
cp $INK2Q_WORK/tf-fork/REVISION $W/tf-revision.txt
# 1. the subset: every held-out sequence's first 512 tokens; teacher and ink2q finals sliced (the model is causal)
$UV run python -m ink2q.tfio subset --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json --max-tokens 512 \
  --finals teacher=$R/teacher/heldout ink2q=$INK2Q_WORK/kortexa-eval/heldout --out $OUT/sub
$UV run python -m ink2q.tfio tokens --tokens $OUT/sub/tokens.safetensors --out $OUT/sub/ids
mkdir -p $OUT/sub/tf-bin
$TF $PACK finals $OUT/sub/ids/index.txt $OUT/sub/tf-bin
$UV run python -m ink2q.tfio finals --src $OUT/sub/tf-bin --out $OUT/sub/tf
for v in ink2q tf; do
  $UV run python -m ink2q.evaluate --source $R/source --tokens $OUT/sub/tokens.safetensors --manifest $OUT/sub/manifest.json \
    --teacher $OUT/sub/teacher --variant $OUT/sub/$v --out $W/sub-$v.json
done
$UV run python -m ink2q.compare $W/sub-ink2q.json $W/sub-tf.json --out $W/sub-compare.json > /dev/null
$UV run python -m ink2q.tfio verdict --compare $W/sub-compare.json --max-hi 0.002 | tee $W/sub-verdict.json
# 2. the full held-out set
$UV run python -m ink2q.tfio tokens --tokens $D/heldout.safetensors --out $OUT/full/ids
mkdir -p $OUT/full/tf-bin
$TF $PACK finals $OUT/full/ids/index.txt $OUT/full/tf-bin
$UV run python -m ink2q.tfio finals --src $OUT/full/tf-bin --out $OUT/full/tf
$UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json \
  --teacher $R/teacher/heldout --variant $OUT/full/tf --out $W/full-tf.json --positions $W/full-tf.npz
$UV run python -m ink2q.compare $R/reports/exl3q/kortexa-heldout.json $W/full-tf.json --out $W/full-compare.json > /dev/null
$UV run python -m ink2q.tfio verdict --compare $W/full-compare.json --max-hi 0.002 | tee $W/full-verdict.json
# 3. decode: a held-out prompt of 512 tokens, then 128 greedy steps
$TF $PACK greedy $OUT/sub/ids/seq-00001.json 128 2>&1 | tee $W/greedy.txt
echo done
