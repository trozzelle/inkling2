#!/usr/bin/env bash
# scripts/studio/tf_decode_gate2.sh: runs inside window.sh on the inkling2-decode fork build
# (tf_sync.sh ../inkling2-tensorfold tf-inkling-run tf-inkling-kernels first; dirty tree refused).
#
# Narrowly scoped to what global shared-KV window attention (ik_attn_part_w) actually touches:
#   0. preflight: inputs, w1 reference logs, clean REVISION, binary usage (--ab);
#   1. (fatal) wall 48 32768 6 100 --ab window_attention (3 runs): digests_equal=true and median saved_ms >= 0.300;
#   2. (fatal) greedy 32k --drafts (64 tokens): 64/64 tokens identical to w1/greedy-32k-drafts.log;
#   3. (fatal) greedy 131k --drafts (64 tokens): 64/64 tokens identical to w1/greedy-131k-drafts.log;
#   4. (diagnostic) wall 48 131072 6 44 --ab window_attention (1 run): full-model 131k saving & digest equality.
#
# Waived (unchanged code paths already gated in w1): exact at 448, layers --window at 512, 8k greedy,
# plain 1-row greedy at 32k/131k, wall at 512 (ik_attn_part_w activates only for global layers with
# (rows >= 3 && blocks >= 96) or (rows >= 2 && blocks >= 256)).
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "$0")/env.sh" && need UV || exit 1

R=$INK2Q_ROOT
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
M3=$INK2Q_WORK/tf-m3
LONG32K=$M3/long32k.json
NEEDLE131K=$M3/needle131k.json
W1=$R/reports/tf-m3/w1
REF32=$W1/greedy-32k-drafts.log
REF131=$W1/greedy-131k-drafts.log
W=$R/reports/tf-m3/w2

unset INKLING_PREFILL_ROWS INKLING_DRAFT_DEPTH INKLING_DRAFT_CONFIDENCE INKLING_DRAFT_BITS INKLING_ROUND_TIMING INKLING_PROMPT_PATH INKLING_DECODE_LEGACY

# 0. preflight
mkdir -p "$W"
for f in "$UV" "$TF" "$TF_FORK/REVISION" "$PACK/config.json" "$PACK/mtp.safetensors" \
         "$LONG32K" "$NEEDLE131K" "$REF32" "$REF131"; do
  [ -e "$f" ] || { echo "preflight: missing $f" >&2; exit 1; }
done
usage=$($TF 2>&1 || true)
case "$usage" in *"--ab NAME"*) ;; *) echo "preflight: $TF wall mode has no --ab NAME" >&2; exit 1 ;; esac
read -r REV DIRTY < "$TF_FORK/REVISION"
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)" >&2; exit 1; }
cp "$TF_FORK/REVISION" "$W/tf-revision.txt"
cd "$STUDIO_REPO"
echo "preflight: pass ($REV $DIRTY)"

FAILED=""
run() {
  local name=$1 rc
  shift
  echo "$(date '+%F %T') == $name: $*"
  set +e
  "$@" 2>&1 | tee "$W/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  return $rc
}
must() {
  run "$@" || { echo "$(date '+%F %T') $1 FAILED: stopping"; exit 1; }
}
leg() {
  run "$@" || { echo "$(date '+%F %T') leg $1 FAILED"; FAILED="$FAILED $1"; return 1; }
}

same_tokens() {
  "$UV" run python - "$1" "$2" "$3" <<'PY'
import json, sys
def tokens(path):
    for line in open(path):
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if isinstance(v, list) and all(isinstance(x, int) for x in v):
            return v
    raise SystemExit(f"no token list in {path}")
a, b, need = tokens(sys.argv[1]), tokens(sys.argv[2]), int(sys.argv[3])
n = min(len(a), len(b))
print(json.dumps({"a": len(a), "b": len(b), "compared": n, "equal": a[:n] == b[:n]}))
sys.exit(0 if n >= need and a[:n] == b[:n] else 1)
PY
}

# 1. 48-layer wall --ab at 32,768 (3 runs; fatal on digests_equal and median saved_ms >= 0.300)
for i in 1 2 3; do
  must "wall-32k-ab-$i" "$TF" "$PACK" wall 48 32768 6 100 1 --ab window_attention
done
must wall-32k-ab-check "$UV" run python - "$W/wall-32k-ab-1.log" "$W/wall-32k-ab-2.log" "$W/wall-32k-ab-3.log" <<'PY'
import json, statistics, sys
rows = []
for p in sys.argv[1:]:
    found = None
    for line in open(p):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and "ab" in d:
            found = d
    assert found is not None, f"no ab json in {p}"
    assert found.get("digests_equal") is True, f"digest mismatch in {p}: {found}"
    rows.append(found)
med_saved = statistics.median(r["saved_ms"] for r in rows)
med_new = statistics.median(r["new_gpu_ms"] for r in rows)
med_leg = statistics.median(r["legacy_gpu_ms"] for r in rows)
out = {
    "runs": len(rows),
    "saved_ms_all": [r["saved_ms"] for r in rows],
    "median_new_gpu_ms": round(med_new, 3),
    "median_legacy_gpu_ms": round(med_leg, 3),
    "median_saved_ms": round(med_saved, 3),
    "pass_keep_rule": med_saved >= 0.3,
}
print(json.dumps(out))
sys.exit(0 if med_saved >= 0.3 else 1)
PY

# 2. 48-layer greedy 32k --drafts (fatal: 64/64 tokens identical to w1 reference)
must greedy-32k-drafts "$TF" "$PACK" greedy "$LONG32K" 64 --drafts
must greedy-32k-same same_tokens "$W/greedy-32k-drafts.log" "$REF32" 64

# 3. 48-layer greedy 131k --drafts (fatal: 64/64 tokens identical to w1 reference)
must greedy-131k-drafts "$TF" "$PACK" greedy "$NEEDLE131K" 64 --drafts
must greedy-131k-same same_tokens "$W/greedy-131k-drafts.log" "$REF131" 64

# 4. (diagnostic) 48-layer wall --ab at 131,072 (1 run: full-model 131k saving & digest equality)
leg wall-131k-ab "$TF" "$PACK" wall 48 131072 6 44 1 --ab window_attention

if [ -n "$FAILED" ]; then
  echo "$(date '+%F %T') decode gate finished with diagnostic failures:$FAILED"
  exit 1
fi
echo "$(date '+%F %T') decode gate: ALL PASS"
