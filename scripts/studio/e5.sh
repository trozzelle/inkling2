#!/usr/bin/env bash
# scripts/studio/e5.sh: runs inside window.sh. Full-layer statistics and FP64 GPTQ for three layers chosen from W2,
# then the splice of GPTQ candidates against RTN and oQe controls; writes gain.json and proj_ratio.json.
set -euo pipefail
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
UV=$UV
D=$R/data
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
E=$R/reports/e5
WORK=$INK2Q_WORK/e5
cd "$REPO"
mkdir -p $E $WORK
( while true; do
    echo "$(date +%T) $(vm_stat | awk '/free|wired|active|inactive|compressor/ {gsub(/\./,"",$NF); printf "%s=%.1f ", $2, $NF*16384/2^30}') swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 5
  done ) > "$E/memory.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
step() { echo "$(date '+%F %T') $*"; }

LAYERS=$($UV run python scripts/e5_choose.py $R/reports/w2/profile/eval $OQE/config.json)
echo "$LAYERS" > $E/layers.txt
step "layers $LAYERS"
if [ ! -f $WORK/receipts.json ]; then
  $UV run --group ref python scripts/e5_layers.py --source $R/source --calib $D/calib.safetensors \
    --subset $R/reports/e2-calib-subset.json --e2 $R/reports/e2.json --oqe $OQE --layers "$LAYERS" \
    --out $WORK --variants $E/variants.json
fi
cp $WORK/receipts.json $E/receipts.json
step splice
$UV run python -m ink2q.harness --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --variants $E/variants.json --out $E/profile --max-per-seq 512 --budget-gib 110
$UV run python -m ink2q.evaluate --source $R/source --tokens $E/profile/tokens.safetensors \
  --manifest $E/profile/manifest.json --teacher $E/profile/teacher --variants-root $E/profile --out-dir $E/profile/eval
$UV run python scripts/e5_summary.py $E/profile/eval "$LAYERS" $E
step done
