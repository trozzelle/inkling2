#!/usr/bin/env bash
# scripts/studio/w2d.sh: runs inside window.sh. Diagnostic for W2's per-layer profile: harness null swap, RTN 8 vs
# RTN 4 at single layers, and leave-one-out marginal sensitivity around all-oQe.
set -euo pipefail
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
UV=$UV
D=$R/data
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
W=$R/reports/w2d
cd "$REPO"
mkdir -p $W
$UV run python scripts/w2d_variants.py $OQE > $W/variants.json
$UV run python -m ink2q.harness --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --variants $W/variants.json --out $W/profile --max-per-seq 512 --budget-gib 110
$UV run python -m ink2q.evaluate --source $R/source --tokens $W/profile/tokens.safetensors \
  --manifest $W/profile/manifest.json --teacher $W/profile/teacher --variants-root $W/profile --out-dir $W/profile/eval
echo done
