#!/usr/bin/env bash
# scripts/studio/gptq_splice.sh: runs inside window.sh. Full-layer statistics and FP64 GPTQ for three layers chosen from the per-layer profile,
# then the splice of GPTQ candidates against RTN and oQe controls; writes gain.json and proj_ratio.json.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
D=$R/data
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
E=$R/reports/e5
WORK=$INK2Q_WORK/e5
cd "$STUDIO_REPO"
mkdir -p $E $WORK
( while true; do
    echo "$(date +%T) $(vm_stat | awk '/free|wired|active|inactive|compressor/ {gsub(/\./,"",$NF); printf "%s=%.1f ", $2, $NF*16384/2^30}') swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 5
  done ) > "$E/memory.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
step() { echo "$(date '+%F %T') $*"; }

LAYERS=$($UV run python scripts/gptq_choose_layers.py $R/reports/w2/profile/eval $OQE/config.json)
echo "$LAYERS" > $E/layers.txt
step "layers $LAYERS"
if [ ! -f $WORK/receipts.json ]; then
  $UV run --group ref python scripts/gptq_splice_variants.py --source $R/source --calib $D/calib.safetensors \
    --subset $R/reports/e2-calib-subset.json --e2 $R/reports/e2.json --oqe $OQE --layers "$LAYERS" \
    --out $WORK --variants $E/variants.json
fi
cp $WORK/receipts.json $E/receipts.json
step splice
$UV run python -m ink2q.harness --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --variants $E/variants.json --out $E/profile --max-per-seq 512 --budget-gib 110
$UV run python -m ink2q.evaluate --source $R/source --tokens $E/profile/tokens.safetensors \
  --manifest $E/profile/manifest.json --teacher $E/profile/teacher --variants-root $E/profile --out-dir $E/profile/eval
$UV run python scripts/gptq_summary.py $E/profile/eval "$LAYERS" $E
step done
