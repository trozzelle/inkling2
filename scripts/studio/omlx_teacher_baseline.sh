#!/usr/bin/env bash
# scripts/studio/omlx_teacher_baseline.sh: runs inside window.sh. Teacher pass, then served oQe fidelity and speed through oMLX.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV STUDIO_HOME OMLX_BIN || exit 1
R=$INK2Q_ROOT
D=$R/data
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
OMLX=$OMLX_BIN   # internal SSD: an exFAT volume's ._ files break venvs
cd "$STUDIO_REPO"
$UV run python -m ink2q.teacher --source $R/source --tokens $D/heldout.safetensors --out $R/teacher/heldout
$UV run python -m ink2q.evaluate --source $R/source --tokens $D/heldout.safetensors \
  --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout --out $R/reports/g1b.json
PYTHONPATH=$STUDIO_REPO $OMLX/python scripts/score_omlx.py --model $OQE --source $R/source \
  --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout \
  --out $R/reports/oqe-heldout.json
KEY=$(openssl rand -hex 16)
echo "$KEY" > $R/reports/oqe.key
# the default guard (balanced) sets a 221.7 GB hard limit and rejects every prompt for this 217.7 GB build
$OMLX/omlx serve --model-dir $R/models --port 8011 --api-key "$KEY" --memory-guard-gb 240 \
  > $R/reports/omlx-serve.log 2>&1 &
SERVE=$!
trap 'kill $SERVE 2>/dev/null; wait $SERVE 2>/dev/null' EXIT
for _ in $(seq 900); do
  kill -0 "$SERVE" 2>/dev/null || { echo "omlx serve exited; see omlx-serve.log" >&2; exit 1; }
  curl -fsS -H "Authorization: Bearer $KEY" http://127.0.0.1:8011/v1/models >/dev/null 2>&1 && break
  sleep 2
done
mkdir -p $R/reports/oqe-speed
# `studio smoke` forces one request with no warmup; `bench` honours the flags. The warmup request absorbs
# the model load, which oMLX defers until the first request.
$STUDIO_HOME/bin/studio bench --base-url http://127.0.0.1:8011/v1 \
  --model Inkling-2-Small-oQe-3.07bpw-MLX --api-key-file $R/reports/oqe.key \
  --concurrency 1 --requests 4 --max-tokens 512 --warmup 1 --timeout 1800 \
  --output $R/reports/oqe-speed/mtp-default-$(date +%Y%m%d-%H%M%S).json
