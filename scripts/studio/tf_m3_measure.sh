#!/usr/bin/env bash
# scripts/studio/tf_m3_measure.sh [--light|--allow-2l]: Phase 0 measurements on the inference Mac.
# Requires an idle GPU (no engine loaded, no request in flight).
# --light runs only the measurements strictly under 12 GiB (launch, read, dense, split, xd, dspark, expert-hist).
# --allow-2l permits the 2-layer load (~20.25 GiB) for wall ablation and prefill --cuts when approved.
set -euo pipefail
. "$(dirname "$0")/env.sh" && need UV

MODE=full
if [ "${1:-}" = "--light" ]; then
  MODE=light
elif [ "${1:-}" = "--allow-2l" ]; then
  MODE=allow2l
elif [ -n "${1:-}" ]; then
  echo "usage: tf_m3_measure.sh [--light|--allow-2l]" >&2
  exit 2
fi

B=$TF_FORK/zig-out/bin/tf-inkling-bench
T=$TF_FORK/zig-out/bin/tf-inkling-run
TAPS=$INK2Q_WORK/tf-m2p0/taps
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
XM=$TAPS/seq-00014.xm.bin
EIDS=$TAPS/seq-00014.eids.bin
OUT=$INK2Q_ROOT/reports/tf-m3/phase0
WORK=$INK2Q_WORK/tf-m3/phase0

mkdir -p "$OUT" "$WORK/dspark"
cd "$STUDIO_REPO"

# Preflight
for f in "$B" "$T" "$TF_PACK/config.json" "$XM" "$EIDS" "$LONG"; do
  [ -e "$f" ] || { echo "missing required input: $f" >&2; exit 1; }
done
if pgrep -fl 'tensorfold-native|omlx|mlx_lm' >/dev/null 2>&1; then
  echo "GPU is not idle (an inference engine is running); refusing to run timings" >&2
  exit 1
fi

# Guard: check load size
if [ "$MODE" = "full" ]; then
  CHECK_OUT=$("$T" "$TF_PACK" check 2 2>&1)
  echo "$CHECK_OUT" | tee "$OUT/check2.log"
  "$UV" run python -c '
import json, sys
for line in sys.argv[1].splitlines():
    if line.startswith("{"):
        d = json.loads(line)
        gib = d["gib"]
        print(f"2-layer check: {gib:.2f} GiB")
        assert gib <= 16.0, f"2-layer load {gib:.2f} GiB exceeds 16 GiB limit"
        sys.exit(0)
sys.exit("no JSON line from check 2")
' "$CHECK_OUT"
elif [ "$MODE" = "light" ]; then
  CHECK_OUT=$("$T" "$TF_PACK" check 0 2>&1)
  echo "$CHECK_OUT" | tee "$OUT/check0.log"
fi

cp "$TF_FORK/REVISION" "$OUT/tf-revision.txt"

# Three interleaved passes
for pass in 1 2 3; do
  LOG="$OUT/pass-$pass.jsonl"
  : > "$LOG"
  echo "=== PASS $pass ($MODE) ===" | tee -a "$LOG"
  "$B" launch 2>&1 | tee -a "$LOG"
  "$B" read 2>&1 | tee -a "$LOG"
  "$B" dense 1@8 8 2>&1 | tee -a "$LOG"
  "$B" split 2>&1 | tee -a "$LOG"
  if [ "$MODE" != "light" ]; then
    "$T" "$TF_PACK" wall 2 512 1 200 24 2>&1 | tee -a "$LOG"
    for c in dense norm attention_prepare attention route experts activation; do
      "$T" "$TF_PACK" wall 2 512 1 200 24 --ablate "$c=stub" 2>&1 | tee -a "$LOG"
      "$T" "$TF_PACK" wall 2 512 1 200 24 --ablate "$c=skip" 2>&1 | tee -a "$LOG"
    done
    "$T" "$TF_PACK" wall 2 512 6 200 24 2>&1 | tee -a "$LOG"
    "$T" "$TF_PACK" prefill "$LONG" --layers 2 --repeat 24 --spans --cuts 4,16,64,256,512 2>&1 | tee -a "$LOG"
  fi
  for l in 0 30; do
    "$B" xd "$TF_PACK" "$l" "$XM" "$EIDS" 1,2,3,4,6,8 parts 2>&1 | tee -a "$LOG"
    "$B" xd "$TF_PACK" "$l" "$XM" "$EIDS" 1,4 parts ablate=rot 2>&1 | tee -a "$LOG"
    "$B" xd "$TF_PACK" "$l" "$XM" "$EIDS" 1,4 parts ablate=decode 2>&1 | tee -a "$LOG"
    for p in 2 3 4; do
      "$B" xd "$TF_PACK" "$l" "$XM" "$EIDS" 1 "shared=$p" 2>&1 | tee -a "$LOG"
    done
  done
done

# Measurement 6: draft head with late context (once, not timed)
echo "=== DSPARK REPLAY ==="
"$T" "$TF_PACK" dspark "$TAPS" 0..20 "$WORK/dspark/full" 2>&1 | tee "$OUT/dspark-full.log"
"$T" "$TF_PACK" dspark "$TAPS" 0..20 "$WORK/dspark/back-1023" --from-back 1023 2>&1 | tee "$OUT/dspark-back-1023.log"
for s in $(seq -f "%05g" 0 20); do
  cmp "$WORK/dspark/full/seq-$s.props.bin" "$WORK/dspark/back-1023/seq-$s.props.bin"
  cmp "$WORK/dspark/full/seq-$s.probs.bin" "$WORK/dspark/back-1023/seq-$s.probs.bin"
done
echo "dspark --from-back 1023 matches full byte-for-byte" | tee "$OUT/dspark-cmp.log"
for K in 4 32 128 512; do
  "$T" "$TF_PACK" dspark "$TAPS" 0..20 "$WORK/dspark/back-$K" --from-back "$K" 2>&1 | tee "$OUT/dspark-back-$K.log"
done

"$UV" run python - "$TAPS" "$WORK/dspark" "$OUT/dspark-from-back.json" <<'PY'
import json, pathlib, sys
import numpy as np
from ink2q.tfaccept import accept_lengths
from ink2q import tfdepth

taps_dir = pathlib.Path(sys.argv[1])
dspark_dir = pathlib.Path(sys.argv[2])
out_path = pathlib.Path(sys.argv[3])

variants = ["full", "back-512", "back-128", "back-32", "back-4"]
horizons = [16, 64, 256, None]
cost = tfdepth.Cost()
results = {}

for v in variants:
    vdir = dspark_dir / v
    results[v] = {}
    for hz in horizons:
        label = "all" if hz is None else f"first_{hz}"
        tally = tfdepth.Tally()
        mean_lens = []
        rows_total = 0
        for seq in range(21):
            meta = json.loads((taps_dir / f"seq-{seq:05d}.tokens.json").read_text())
            prompt = meta["prompt"]
            tokens = np.asarray(meta["tokens"], dtype=np.int64)
            T = len(tokens) - 1
            props = np.fromfile(vdir / f"seq-{seq:05d}.props.bin", dtype=np.int32).reshape(T, 7)
            probs = np.fromfile(vdir / f"seq-{seq:05d}.probs.bin", dtype=np.float32).reshape(T, 7)
            lo = prompt - 1
            hi = T if hz is None else min(T, lo + hz)
            if hi <= lo:
                continue
            al = accept_lengths(props, tokens, lo, hi)
            kept = tfdepth.kept_floor(probs[lo:hi], 0.70)
            tally.add(tfdepth.walk(al, kept))
            mean_lens.append(float(al.mean()))
            rows_total += hi - lo
        entry = tally.entry(cost)
        entry["rows"] = rows_total
        entry["mean_accept_len"] = float(np.mean(mean_lens)) if mean_lens else 0.0
        results[v][label] = entry

out_path.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
PY

# Measurement 7 helper: expert sharing histogram over all 21 sequences' eids.bin
"$UV" run python - "$TAPS" "$OUT/expert-hist.jsonl" <<'PY'
import json, pathlib, sys
import numpy as np

taps_dir = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])

# Load generated rows of all 21 sequences
seqs = []
for seq in range(21):
    meta = json.loads((taps_dir / f"seq-{seq:05d}.tokens.json").read_text())
    prompt = meta["prompt"]
    T = len(meta["tokens"]) - 1
    eids = np.fromfile(taps_dir / f"seq-{seq:05d}.eids.bin", dtype=np.uint32).reshape(T, 48, 8)
    seqs.append(eids[prompt - 1:])

lines = []
for R in range(2, 9):
    for L in range(48):
        hist = np.zeros(R + 1, dtype=np.int64)
        windows = 0
        for arr in seqs:
            n = arr.shape[0]
            # non-overlapping or stride-1 windows; stride-1 over all generated positions
            for i in range(0, n - R + 1):
                w = arr[i:i + R, L, :].ravel()
                _, cnts = np.unique(w, return_counts=True)
                for c in cnts:
                    hist[min(int(c), R)] += 1
                windows += 1
        rec = {
            "rows": R,
            "layer": L,
            "windows": windows,
            "experts_with_m": [int(x) for x in hist[1:]],
            "mean_experts_with_m": [round(float(x) / max(windows, 1), 4) for x in hist[1:]],
        }
        lines.append(json.dumps(rec))

out_path.write_text("\n".join(lines) + "\n")
print(f"wrote {len(lines)} histogram lines to {out_path}")
PY

echo "PHASE0_STUDIO_DONE"
