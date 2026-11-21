#!/usr/bin/env bash
# scripts/studio/tf_sync.sh FORK_DIR [TARGETS...]: copy the TensorFold fork to the inference Mac and build TARGETS there
# (default: the Inkling programs) with the Zig 0.17.0 bundled with the installed TensorFold runtime.
# The repo must already be synced there (sync.sh): the build sources its scripts/studio/env.sh.
set -euo pipefail
. "$(dirname "$0")/env.sh"
FORK=${1:?usage: tf_sync.sh FORK_DIR [TARGETS...]}
shift
TARGETS=${*:-tf-inkling-run tf-inkling-kernels}
REV=$(git -C "$FORK" rev-parse HEAD)
DIRTY=$(git -C "$FORK" status --porcelain | wc -l | tr -d ' ')
rsync -a --delete --exclude /.git --exclude /.zig-cache --exclude /zig-out "$FORK"/ "$STUDIO_HOST:$TF_FORK/"
ssh "$STUDIO_HOST" "echo $REV dirty=$DIRTY > '$TF_FORK/REVISION'"
ssh "$STUDIO_HOST" STUDIO_REPO="'$STUDIO_REPO'" TARGETS="'$TARGETS'" bash -s <<'REMOTE'
set -euo pipefail
cd "$STUDIO_REPO"
. scripts/studio/env.sh
need UV STUDIO_HOME
export PATH=$(dirname "$UV"):$PATH
ZIG=$(find "$STUDIO_HOME" -maxdepth 6 -type f -path '*zig-aarch64-macos-0.17.0/zig' | head -1)
[ -x "$ZIG" ] || { echo "no Zig 0.17.0 under $STUDIO_HOME" >&2; exit 1; }
JPEG=$(find "$STUDIO_HOME" -maxdepth 7 -name libturbojpeg.0.dylib | head -1)
cd "$TF_FORK"
TENSORFOLD_LIBTURBOJPEG="$JPEG" "$ZIG" build $TARGETS -Doptimize=ReleaseFast
cat REVISION
REMOTE
