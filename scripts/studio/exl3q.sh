#!/usr/bin/env bash
# scripts/studio/exl3q.sh: runs inside window.sh. The kortexa EXL3 2.59 bpw experts, decoded to bf16, in the ink2q
# forward with source non-experts (as B' does for oQe), on the held-out, agentic and long sets; paired comparisons
# against B' and against served oQe.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
D=$R/data
S=$D/sessions
K=$INK2Q_WORK/kortexa
OUT=$INK2Q_WORK/kortexa-eval
W=$R/reports/exl3q
cd "$STUDIO_REPO"
mkdir -p $W
( while true; do
    echo "$(date +%T) $(vm_stat | awk '/free|wired|active|inactive|compressor/ {gsub(/\./,"",$NF); printf "%s=%.1f ", $2, $NF*16384/2^30}') swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 5
  done ) > "$W/memory.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
if [ ! -f $OUT/long/done.json ]; then
  $UV run python -m ink2q.teacher --source $R/source --variant-kortexa $K \
    --tokens $D/heldout.safetensors $S/agentic.safetensors $S/long.safetensors \
    --out $OUT/heldout $OUT/agentic $OUT/long
fi
$UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout --variant $OUT/heldout \
  --out $W/kortexa-heldout.json --positions $W/kortexa-heldout.npz
for s in agentic long; do
  $UV run python -m ink2q.evaluate --source $R/source --tokens $S/$s.safetensors --manifest $S/${s}_manifest.json \
    --teacher $R/teacher/$s --variant $OUT/$s --out $W/kortexa-$s.json --positions $W/kortexa-$s.npz
done
for s in heldout agentic long; do
  $UV run python -m ink2q.compare $R/reports/w2/bprime-$s.json $W/kortexa-$s.json --out $W/bprime-vs-kortexa-$s.json > /dev/null
done
$UV run python -m ink2q.compare $R/reports/oqe-heldout.json $W/kortexa-heldout.json --out $W/oqe-served-vs-kortexa-heldout.json > /dev/null
echo done
