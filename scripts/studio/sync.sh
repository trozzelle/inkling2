#!/usr/bin/env bash
# scripts/studio/sync.sh: copy this repo to the inference Mac (no git remote there; local.env travels with the tree)
# and sync its environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
. scripts/studio/env.sh
rsync -a --delete --exclude /.git --exclude /.venv --exclude /docs/plans --exclude /data --exclude /work \
  --exclude /reports ./ "$STUDIO_HOST:$STUDIO_REPO/"
ssh "$STUDIO_HOST" STUDIO_REPO="'$STUDIO_REPO'" bash -s <<'REMOTE'
set -euo pipefail
cd "$STUDIO_REPO"
. scripts/studio/env.sh
need UV
"$UV" sync -q
"$UV" run pytest -q
REMOTE
