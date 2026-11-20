#!/usr/bin/env bash
# scripts/studio/tf_sync.sh FORK_DIR [TARGETS...]: copy the TensorFold fork to the inference Mac and build TARGETS there
# (default: the Inkling programs) with the Zig 0.17.0 that the studio-inference TensorFold runtime ships.
set -euo pipefail
FORK=${1:?usage: tf_sync.sh FORK_DIR [TARGETS...]}
shift
TARGETS=${*:-tf-inkling-run tf-inkling-kernels}
REV=$(git -C "$FORK" rev-parse HEAD)
DIRTY=$(git -C "$FORK" status --porcelain | wc -l | tr -d ' ')
rsync -a --delete --exclude /.git --exclude /.zig-cache --exclude /zig-out "$FORK"/ "$STUDIO_HOST":$INK2Q_WORK/tf-fork/
ssh "$STUDIO_HOST" "echo $REV dirty=$DIRTY > $INK2Q_WORK/tf-fork/REVISION"
ssh "$STUDIO_HOST" TARGETS="'$TARGETS'" 'bash -s' <<'EOF'
set -euo pipefail
export PATH=$(dirname "$UV"):$PATH
ZIG=$(find $STUDIO_HOME -maxdepth 6 -type f -path '*zig-aarch64-macos-0.17.0/zig' | head -1)
[ -x "$ZIG" ] || { echo "no Zig 0.17.0 under $STUDIO_HOME" >&2; exit 1; }
JPEG=$(find $STUDIO_HOME -maxdepth 7 -name libturbojpeg.0.dylib | head -1)
cd $INK2Q_WORK/tf-fork
TENSORFOLD_LIBTURBOJPEG="$JPEG" "$ZIG" build $TARGETS -Doptimize=ReleaseFast
cat REVISION
EOF
