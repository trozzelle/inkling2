#!/usr/bin/env bash
# scripts/studio/tf_decode_fidelity.sh: runs inside window.sh. The fidelity gates with every row past the first 16 fed in
# 16-row windows through the decode path: the layer gate over 48 layers against the existing reference and floor, then
# TensorFold's finals over the full held-out set scored against the teacher and compared with ink2q's kortexa result.
set -uo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
D=$R/data
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
L=$R/reports/tf-m1/layers
IDS=$INK2Q_WORK/tf-m1/full/ids
OUT=$INK2Q_WORK/tf-m2p1
W=$R/reports/tf-m2p1
REF=$R/reports/exl3q/kortexa-heldout.json
set -e
mkdir -p $OUT/full/tf-bin $W
cd "$STUDIO_REPO"
cp $TF_FORK/REVISION $W/tf-revision-fidelity.txt
# 1. layer gate, window mode (the reference, floor and ids from the earlier layer gate)
$TF $PACK layers $L/ids.json $OUT/layers-win.bin 48 --window
$UV run python -m ink2q.tflayers gate --ref $L --tf $OUT/layers-win.bin --out $W/layer-gate-window.json
# 2. full held-out finals, window mode
[ -f $IDS/index.txt ] || $UV run python -m ink2q.tfio tokens --tokens $D/heldout.safetensors --out $IDS
$TF $PACK finals $IDS/index.txt $OUT/full/tf-bin --window
$UV run python -m ink2q.tfio finals --src $OUT/full/tf-bin --out $OUT/full/tf
$UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json \
  --teacher $R/teacher/heldout --variant $OUT/full/tf --out $W/full-tf-window.json --positions $W/full-tf-window.npz
$UV run python -m ink2q.compare $REF $W/full-tf-window.json --out $W/full-compare-tf-window.json > /dev/null
set +e
$UV run python -m ink2q.tfio verdict --compare $W/full-compare-tf-window.json --max-hi 0.002 | tee $W/full-verdict-tf-window.json
echo done
