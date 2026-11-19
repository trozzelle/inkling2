#!/usr/bin/env bash
# scripts/studio/sync.sh: copy this repo to the inference Mac (no git remote there) and sync its environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
rsync -a --delete --exclude /.git --exclude /.venv --exclude /docs/plans --exclude /data --exclude /work \
  --exclude /reports ./ "$STUDIO_HOST":$STUDIO_REPO/
ssh "$STUDIO_HOST" 'cd $STUDIO_REPO && $UV sync -q && $UV run pytest -q'
