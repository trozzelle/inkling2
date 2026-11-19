#!/usr/bin/env bash
# scripts/studio/w1b.sh: runs inside window.sh. The oQe half of W1 (baseline B) with system memory sampled
# every 5 s, for when the teacher half already succeeded.
set -euo pipefail
R=$INK2Q_ROOT
REPO=$STUDIO_REPO
D=$R/data
OQE=$R/models/Inkling-2-Small-oQe-3.07bpw-MLX
OMLX=$OMLX_BIN
cd "$REPO"
( while true; do
    echo "$(date +%T) $(vm_stat | awk '/free|wired|active|inactive|compressor/ {gsub(/\./,"",$NF); printf "%s=%.1f ", $2, $NF*16384/2^30}') swap=$(sysctl -n vm.swapusage | awk '{print $6}')"
    sleep 5
  done ) > "$R/reports/w1b-memory.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
if [ ! -s $R/reports/oqe-heldout.json ]; then
  PYTHONPATH=$REPO $OMLX/python scripts/score_omlx.py --model $OQE --source $R/source \
    --tokens $D/heldout.safetensors --manifest $D/heldout_manifest.json --teacher $R/teacher/heldout \
    --out $R/reports/oqe-heldout.json
fi
KEY=$(openssl rand -hex 16)
echo "$KEY" > $R/reports/oqe.key
# the default guard (balanced) sets a 221.7 GB hard limit and rejects every prompt for this 217.7 GB build
$OMLX/omlx serve --model-dir $R/models --port 8011 --api-key "$KEY" --memory-guard-gb 240 \
  > $R/reports/omlx-serve.log 2>&1 &
SERVE=$!
trap 'kill $SAMPLER $SERVE 2>/dev/null; wait $SERVE 2>/dev/null' EXIT
for _ in $(seq 900); do
  kill -0 "$SERVE" 2>/dev/null || { echo "omlx serve exited; see omlx-serve.log" >&2; exit 1; }
  curl -fsS -H "Authorization: Bearer $KEY" http://127.0.0.1:8011/v1/models >/dev/null 2>&1 && break
  sleep 2
done
mkdir -p $R/reports/oqe-speed
$STUDIO_HOME/bin/studio bench --base-url http://127.0.0.1:8011/v1 \
  --model Inkling-2-Small-oQe-3.07bpw-MLX --api-key-file $R/reports/oqe.key \
  --concurrency 1 --requests 4 --max-tokens 512 --warmup 1 --timeout 1800 \
  --output $R/reports/oqe-speed/mtp-default-$(date +%Y%m%d-%H%M%S).json
