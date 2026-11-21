#!/usr/bin/env bash
# scripts/studio/w2.sh: runs inside window.sh. Teacher finals for the session sets, the fair oQe baseline B'
# (oQe experts, source non-experts, ink2q forward), the per-layer profile and the NVFP4 yardstick.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
D=$R/data
S=$D/sessions
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
W=$R/reports/w2
cd "$STUDIO_REPO"
mkdir -p $W
( while true; do
    echo "$(date +%T) $(vm_stat | awk '/free|wired|active|inactive|compressor/ {gsub(/\./,"",$NF); printf "%s=%.1f ", $2, $NF*16384/2^30}') swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 5
  done ) > "$W/memory.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
step() { echo "$(date '+%F %T') $*"; }

# 1. teacher finals for the agentic and long session sets (held-out teacher from W1 is reused)
if [ ! -f $R/teacher/long/done.json ]; then
  step teacher
  $UV run python -m ink2q.teacher --source $R/source --tokens $S/agentic.safetensors $S/long.safetensors \
    --out $R/teacher/agentic $R/teacher/long
fi

# 2. B': oQe experts installed in the ink2q forward, all three sets in one pass
if [ ! -f $R/bprime/long/done.json ]; then
  step bprime
  $UV run python -m ink2q.teacher --source $R/source --variant-mlx $OQE \
    --tokens $D/heldout.safetensors $S/agentic.safetensors $S/long.safetensors \
    --out $R/bprime/heldout $R/bprime/agentic $R/bprime/long
fi

# 3. score B' per set with per-position records; compare served B against B' on the held-out set
step evaluate-bprime
$UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout --variant $R/bprime/heldout \
  --out $W/bprime-heldout.json --positions $W/bprime-heldout.npz
for s in agentic long; do
  $UV run python -m ink2q.evaluate --source $R/source --tokens $S/$s.safetensors \
    --manifest $S/${s}_manifest.json --teacher $R/teacher/$s --variant $R/bprime/$s \
    --out $W/bprime-$s.json --positions $W/bprime-$s.npz
done
$UV run python -m ink2q.compare $R/reports/oqe-heldout.json $W/bprime-heldout.json --out $W/b-vs-bprime.json

# 4. per-layer profile and NVFP4 yardstick through the splice harness (held-out, 512 tokens per sequence)
step profile
$UV run python scripts/w2_variants.py $OQE > $W/variants.json
$UV run python -m ink2q.harness --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --variants $W/variants.json --out $W/profile --max-per-seq 512 --budget-gib 110
step evaluate-profile
$UV run python -m ink2q.evaluate --source $R/source --tokens $W/profile/tokens.safetensors \
  --manifest $W/profile/manifest.json --teacher $W/profile/teacher --variants-root $W/profile \
  --out-dir $W/profile/eval
step done
