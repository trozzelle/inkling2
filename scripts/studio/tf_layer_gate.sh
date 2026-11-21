#!/usr/bin/env bash
# scripts/studio/tf_layer_gate.sh: runs inside window.sh. The ink2q reference and its fp32-convolution floor with the
# kortexa experts over 48 layers of one held-out sequence, the TensorFold dump of the same, and the layer gate.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
K=$INK2Q_WORK/kortexa
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
W=$R/reports/tf-m1
mkdir -p $W/layers
cd "$STUDIO_REPO"
cp $TF_FORK/REVISION $W/tf-revision.txt
if [ ! -f $W/layers/floor.npy ]; then
  $UV run python -m ink2q.tflayers ref --source $R/source --kortexa $K --tokens $R/data/heldout.safetensors \
    --manifest $R/data/heldout_manifest.json --case en_web --max-tokens 512 --layers 48 --out $W/layers
fi
$TF $PACK check
$TF $PACK layers $W/layers/ids.json $W/layers/tf.bin 48
$UV run python -m ink2q.tflayers gate --ref $W/layers --tf $W/layers/tf.bin --out $W/layer-gate.json
$TF $PACK exact $W/layers/ids.json 448
echo done
