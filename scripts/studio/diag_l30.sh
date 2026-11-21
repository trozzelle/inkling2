#!/usr/bin/env bash
# scripts/studio/diag_l30.sh: runs inside window.sh. Saves layer-30 Grams for experts 16-31 for an offline GPTQ replay.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV || exit 1
R=$INK2Q_ROOT
cd "$STUDIO_REPO"
$UV run python scripts/diag_l30_stats.py --source $R/source --calib $R/data/calib.safetensors \
  --subset $R/reports/e2-calib-subset.json --layer 30 --experts 16-31 --out $INK2Q_WORK/diag
echo done
