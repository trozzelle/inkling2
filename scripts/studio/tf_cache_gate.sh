#!/usr/bin/env bash
# scripts/studio/tf_cache_gate.sh [--prep-only | RUNTIME_ID]
#
# Modes:
#   tf_cache_gate.sh --prep-only
#     Runs on the Studio outside any downtime window (no model load). Verifies the M2 Phase 3 identity reference
#     files exist, builds $INK2Q_WORK/tf-m3/{long-ids.json,needle131k.json,long32k.json} and the two-document
#     multi-turn prompt set $INK2Q_WORK/tf-m3/turns/prefill-messages.json (~8k and ~32k).
#
#   tf_cache_gate.sh RUNTIME_ID
#     Runs inside window.sh on the fork commit's build (tf_sync.sh FORK_DIR first; RUNTIME_ID names the installed
#     runtime of that commit). Logs land in $INK2Q_ROOT/reports/tf-m3/w1/, bulk outputs in $INK2Q_WORK/tf-m3/w1/.
#     Legs (fatal unless marked):
#       0. preflight: inputs, reference files, binaries, clean REVISION matching RUNTIME_ID;
#       1. exact on 48 layers (differ: 0);
#       2. 48-layer identity dumps (--chunk 4096 and --window), cmp with M2 Phase 3 references;
#       3. wrap identity on the 8k prompt: prefill --drafts h_digest and greedy 64 plain/drafted match references;
#       4. partition --drafts with --restore across wraps and tiles: identical everywhere;
#       5. long contexts: greedy 32k plain/drafted equal; prefill 131k --drafts; greedy 131k plain/drafted equal and
#          needle 7391-KITE recovered (tfchat contains);
#       6. served cache-on vs cache-off on a spare port: startup caches fit >= 262,144 tokens, tfrouter turns on both
#          documents, tfrouter same equal, cached_tokens > 0 on follow/regenerate/edit and 32k reask;
#       7. (diagnostic, non-fatal) decode_tok_s ratio (cache-on / cache-off) and ttft_s by op, startup lines, and
#          footprint of the cache-on server after the 32k conversation.
set -euo pipefail
# shellcheck source=env.sh
. "$(dirname "$0")/env.sh"

R=$INK2Q_ROOT
D=$R/data
PACK=$TF_PACK
TF=$TF_FORK/zig-out/bin/tf-inkling-run
L=$R/reports/tf-m1/layers
LONG=$INK2Q_WORK/tf-m2p1-long8k.json
M3=$INK2Q_WORK/tf-m3
LONG_IDS=$M3/long-ids.json
LONG32K=$M3/long32k.json
NEEDLE131K=$M3/needle131k.json
TURNS_DIR=$M3/turns
TURNS_MSGS=$TURNS_DIR/prefill-messages.json
REF_WORK=$INK2Q_WORK/tf-m2p3/w2
REF_REP=$R/reports/tf-m2p3/w2
REF_CHUNK=$REF_WORK/layers-chunk.bin
REF_WINDOW=$REF_WORK/layers-window.bin
REF_GREEDY_PLAIN=$REF_REP/greedy-plain.log
REF_GREEDY_DRAFTS=$REF_REP/greedy-drafts.log
REF_PREFILL=$REF_REP/prefill-drafts-c4096.log
W=$R/reports/tf-m3/w1
OUT=$M3/w1
NEEDLE_CODE="7391-KITE"

check_refs() {
  local f
  for f in "$REF_CHUNK" "$REF_WINDOW" "$REF_GREEDY_PLAIN" "$REF_GREEDY_DRAFTS" "$REF_PREFILL"; do
    if [ ! -e "$f" ]; then
      echo "missing identity reference: $f (stop: baseline build needed on the Studio)" >&2
      exit 1
    fi
  done
}

ids_len() {  # ids_len FILE: count of ints in a JSON int-list file
  python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
assert isinstance(d, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in d)
print(len(d))' "$1" 2>/dev/null || true
}

if [ "${1:-}" = "--prep-only" ]; then
  need UV || exit 1
  for f in "$UV" "$R/source" "$D/heldout.safetensors" "$D/heldout_manifest.json"; do
    [ -e "$f" ] || { echo "prep: missing $f" >&2; exit 1; }
  done
  check_refs
  mkdir -p "$M3" "$TURNS_DIR/doc8k" "$TURNS_DIR/doc32k"
  cd "$STUDIO_REPO" || exit 1

  echo "$(date '+%F %T') == prep: building $LONG_IDS from heldout sequences in manifest order"
  "$UV" run python - "$D/heldout.safetensors" "$D/heldout_manifest.json" "$LONG_IDS" "$TURNS_DIR/doc32k-src.json" <<'PY'
import json, sys
from pathlib import Path
from ink2q.tokens import load_manifest, load_tokens

seqs = load_tokens(Path(sys.argv[1]))
man = load_manifest(Path(sys.argv[2]))
assert len(seqs) == len(man), f"sequence/manifest length mismatch: {len(seqs)} != {len(man)}"
base = [int(t) for s in seqs for t in s]
assert base, "heldout token list is empty"
# Tile heldout sequences so 131k needle prompts have enough source tokens while keeping doc8k and doc32k disjoint.
reps = (160_000 + len(base) - 1) // len(base)
ids = base * reps
Path(sys.argv[3]).write_text(json.dumps(ids))
offset = 16384
Path(sys.argv[4]).write_text(json.dumps(ids[offset:]))
print(json.dumps({"sequences": len(seqs), "base_tokens": len(base), "tiled_tokens": len(ids), "doc32k_offset": offset, "out": sys.argv[3]}))
PY

  echo "$(date '+%F %T') == prep: building $NEEDLE131K and $LONG32K"
  "$UV" run python -m ink2q.tfchat needle --source "$R/source" --ids "$LONG_IDS" \
    --target 131008 --at 0.5 --code "$NEEDLE_CODE" --out "$NEEDLE131K"
  "$UV" run python -m ink2q.tfchat needle --source "$R/source" --ids "$LONG_IDS" \
    --target 32704 --at 0.5 --code "$NEEDLE_CODE" --out "$LONG32K"

  echo "$(date '+%F %T') == prep: building $TURNS_MSGS (doc8k and doc32k)"
  "$UV" run python -m ink2q.tfchat documents --source "$R/source" --target 8192 \
    --doc "doc8k=ids:$LONG_IDS" --out-dir "$TURNS_DIR/doc8k"
  "$UV" run python -m ink2q.tfchat documents --source "$R/source" --target 32768 \
    --doc "doc32k=ids:$TURNS_DIR/doc32k-src.json" --out-dir "$TURNS_DIR/doc32k"

  "$UV" run python - "$TURNS_DIR/doc8k/prefill-messages.json" "$TURNS_DIR/doc32k/prefill-messages.json" "$TURNS_MSGS" <<'PY'
import json, sys
from pathlib import Path

d8 = json.loads(Path(sys.argv[1]).read_text())
d32 = json.loads(Path(sys.argv[2]).read_text())
merged = []
for seq, entry in enumerate(d8 + d32):
    merged.append({**entry, "seq": seq})
Path(sys.argv[3]).write_text(json.dumps(merged, ensure_ascii=False, indent=1) + "\n")
print(json.dumps({"documents": [(e["case"], e.get("ids_length"), e.get("rendered_length")) for e in merged], "out": sys.argv[3]}))
PY
  echo "prep: pass"
  exit 0
fi

need UV STUDIO_HOME STUDIO_ALLOWED_HOSTS || exit 1
RID=${1:?usage: tf_cache_gate.sh [--prep-only | RUNTIME_ID [--from-leg 1|4b|5|6]]}
FROM_LEG=1
if [ "${2:-}" = "--from-leg" ]; then
  FROM_LEG=${3:?--from-leg requires a value (1, 4b, 5, 6)}
fi
BIN=$STUDIO_HOME/runtimes/tensorfold/$RID/source/zig-out/native/bin/tensorfold-native
export TENSORFOLD_LIBTURBOJPEG=$STUDIO_HOME/runtimes/tensorfold/$RID/libjpeg-turbo/lib/libturbojpeg.0.dylib
PORT=19013

unset INKLING_PREFILL_ROWS INKLING_DRAFT_DEPTH INKLING_DRAFT_CONFIDENCE INKLING_DRAFT_BITS INKLING_ROUND_TIMING INKLING_PROMPT_PATH

SERVER_PID=""
cleanup() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap 'cleanup' EXIT

# 0. preflight
mkdir -p "$W" "$OUT"
for f in "$UV" "$TF" "$TF_FORK/REVISION" "$BIN" "$TENSORFOLD_LIBTURBOJPEG" \
         "$PACK/config.json" "$PACK/mtp.safetensors" "$L/ids.json" "$R/source" \
         "$LONG" "$LONG32K" "$NEEDLE131K" "$TURNS_MSGS"; do
  [ -e "$f" ] || { echo "preflight: missing $f" >&2; exit 1; }
done
[ -x "$BIN" ] || { echo "preflight: not executable: $BIN" >&2; exit 1; }
check_refs

[ "$(ids_len "$LONG")" = 8192 ] || { echo "preflight: $LONG is not 8,192 token ids" >&2; exit 1; }
[ "$(ids_len "$LONG32K")" = 32704 ] || { echo "preflight: $LONG32K is not 32,704 token ids" >&2; exit 1; }
[ "$(ids_len "$NEEDLE131K")" = 131008 ] || { echo "preflight: $NEEDLE131K is not 131,008 token ids" >&2; exit 1; }

usage=$("$TF" 2>&1 || true)
for want in "partition <ids.json>" "prefill <ids.json>" "greedy <ids.json>" "--restore"; do
  case "$usage" in
    *"$want"*) ;;
    *) echo "preflight: $TF usage missing '$want'; rebuild with tf_sync.sh" >&2; exit 1 ;;
  esac
done
grep -q 'layers <ids.json>.*--chunk' <<< "$usage" || { echo "preflight: $TF layers mode missing --chunk" >&2; exit 1; }

read -r REV DIRTY < "$TF_FORK/REVISION"
[ "$DIRTY" = "dirty=0" ] || { echo "preflight: tf-fork build is from a dirty tree ($DIRTY)" >&2; exit 1; }
case "$RID" in
  *"${REV:0:12}"*) ;;
  *) echo "preflight: runtime $RID does not match commit ${REV:0:12}" >&2; exit 1 ;;
esac
cp "$TF_FORK/REVISION" "$W/tf-revision.txt"
echo "$RID" > "$W/runtime-id.txt"
cd "$STUDIO_REPO" || exit 1
echo "preflight: pass"

FAILED=""
run() {
  local name=$1 rc
  shift
  echo "$(date '+%F %T') == $name: $*"
  set +e
  "$@" 2>&1 | tee "$W/$name.log"
  rc=${PIPESTATUS[0]}
  set -e
  return "$rc"
}
must() {
  run "$@" || { echo "$(date '+%F %T') $1 FAILED: stopping" >&2; exit 1; }
}
leg() {
  run "$@" || { echo "$(date '+%F %T') leg $1 FAILED" >&2; FAILED="$FAILED $1"; return 1; }
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
        if isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
            return v
    raise SystemExit(f"no token list in {path}")
a, b, need = tokens(sys.argv[1]), tokens(sys.argv[2]), int(sys.argv[3])
n = min(len(a), len(b))
print(json.dumps({"a": len(a), "b": len(b), "compared": n, "equal": a[:n] == b[:n]}))
sys.exit(0 if n >= need and a[:n] == b[:n] else 1)
PY
}

h_digest() {
  "$UV" run python - "$1" <<'PY'
import json, sys
text = open(sys.argv[1]).read()
for line in text.splitlines():
    try:
        v = json.loads(line)
    except ValueError:
        continue
    if isinstance(v, dict) and "h_digest" in v:
        print(v["h_digest"])
        sys.exit(0)
raise SystemExit(f"no h_digest in {sys.argv[1]}")
PY
}

check_partition() {
  "$UV" run python - "$1" "$2" <<'PY'
import json, re, sys
text = open(sys.argv[1]).read()
want = int(sys.argv[2])
obj = None
for line in text.splitlines():
    # Zig `{any}` formats slices with `{ ... }` instead of JSON `[ ... ]`.
    fixed = re.sub(r'"first_cuts":\s*\{([^}]*)\}', r'"first_cuts": [\1]', line)
    try:
        v = json.loads(fixed)
    except ValueError:
        continue
    if isinstance(v, dict) and "partition" in v:
        obj = v
        break
assert obj is not None, f"no partition JSON in {sys.argv[1]}"
assert obj.get("partition") == "identical", f"partition={obj.get('partition')!r}"
restores = obj.get("restore") or []
assert len(restores) == want, f"expected {want} restore legs, got {len(restores)}"
bad = [r for r in restores if r.get("result") != "identical"]
assert not bad, f"non-identical restores: {bad}"
print(json.dumps({"partition": obj["partition"], "restores": len(restores)}))
PY
}

if [ "$FROM_LEG" = 1 ]; then
# 1. exact on 48 layers
must exact "$TF" "$PACK" exact "$L/ids.json" 448
grep -q '"differ": 0,' "$W/exact.log" || { echo "exact: differ is not 0" >&2; exit 1; }

# 2. identity, every layer (cmp with M2 Phase 3 reference dumps)
must layers-chunk "$TF" "$PACK" layers "$L/ids.json" "$OUT/layers-chunk.bin" 48 --chunk 4096
cmp "$OUT/layers-chunk.bin" "$REF_CHUNK" || { echo "layers-chunk: differs from reference" >&2; exit 1; }
must layers-window "$TF" "$PACK" layers "$L/ids.json" "$OUT/layers-window.bin" 48 --window
cmp "$OUT/layers-window.bin" "$REF_WINDOW" || { echo "layers-window: differs from reference" >&2; exit 1; }
echo "layers identity: pass"

# 3. identity across the wrap (8k prompt)
must prefill-8k-drafts "$TF" "$PACK" prefill "$LONG" --spans --drafts --chunk 4096
got_digest=$(h_digest "$W/prefill-8k-drafts.log")
ref_digest=$(h_digest "$REF_PREFILL")
echo "prefill-8k h_digest: got=$got_digest ref=$ref_digest"
[ "$got_digest" = "$ref_digest" ] || { echo "prefill-8k: h_digest differs from reference" >&2; exit 1; }
must greedy-8k-plain "$TF" "$PACK" greedy "$LONG" 64
same_tokens "$W/greedy-8k-plain.log" "$REF_GREEDY_PLAIN" 64 || { echo "greedy-8k-plain: differs from reference" >&2; exit 1; }
must greedy-8k-drafts "$TF" "$PACK" greedy "$LONG" 64 --drafts
same_tokens "$W/greedy-8k-drafts.log" "$REF_GREEDY_DRAFTS" 64 || { echo "greedy-8k-drafts: differs from reference" >&2; exit 1; }
echo "wrap identity: pass"

# 4a. partition with --restore and --drafts (cut 4095 1)
must partition-c4095-1 "$TF" "$PACK" partition "$LONG" 4095 1 --restore 1000 --restore 5377 --restore 8000 --drafts
fi

if [ "$FROM_LEG" = 1 ] || [ "$FROM_LEG" = 4b ]; then
check_partition "$W/partition-c4095-1.log" 3 || { echo "partition-c4095-1: not identical everywhere" >&2; exit 1; }
# 4b. partition with --restore and --drafts (cut 2053)
must partition-tile "$TF" "$PACK" partition "$LONG" 2053 --restore 4096 --drafts
check_partition "$W/partition-tile.log" 1 || { echo "partition-tile: not identical everywhere" >&2; exit 1; }
echo "partition restore: pass"
fi

if [ "$FROM_LEG" = 1 ] || [ "$FROM_LEG" = 4b ] || [ "$FROM_LEG" = 5 ]; then
# 5. long contexts (32k and 131k needle)
must greedy-32k-plain "$TF" "$PACK" greedy "$LONG32K" 64
must greedy-32k-drafts "$TF" "$PACK" greedy "$LONG32K" 64 --drafts
same_tokens "$W/greedy-32k-plain.log" "$W/greedy-32k-drafts.log" 64 || { echo "greedy-32k: drafted differs from plain" >&2; exit 1; }

must prefill-131k-drafts "$TF" "$PACK" prefill "$NEEDLE131K" --spans --drafts --chunk 4096

must greedy-131k-plain "$TF" "$PACK" greedy "$NEEDLE131K" 64
must greedy-131k-drafts "$TF" "$PACK" greedy "$NEEDLE131K" 64 --drafts
same_tokens "$W/greedy-131k-plain.log" "$W/greedy-131k-drafts.log" 64 || { echo "greedy-131k: drafted differs from plain" >&2; exit 1; }
must needle-131k-contains "$UV" run python -m ink2q.tfchat contains \
  --source "$R/source" --log "$W/greedy-131k-plain.log" --text "$NEEDLE_CODE"
echo "long contexts: pass"
fi

# 6. served: cache on against cache off
read -ra _host_list <<< "$STUDIO_ALLOWED_HOSTS"
HOSTS=()
for h in "${_host_list[@]}"; do
  HOSTS+=(--allowed-host "$h")
done

serve_turns() {
  local name=$1
  shift
  local log=$W/server-$name.log
  local out_json=$W/turns-$name.json
  local t0 code ready=0 i
  t0=$(date +%s)
  echo "$(date '+%F %T') == serve-$name: starting tensorfold-native on 127.0.0.1:$PORT"
  "$BIN" serve "$PACK" --name inkling2 --host 127.0.0.1 --port "$PORT" \
    --context 65536 --parallel 1 --load-limit-gb 236 "${HOSTS[@]}" "$@" > "$log" 2>&1 &
  SERVER_PID=$!

  for ((i = 1; i <= 96; i++)); do
    sleep 5
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "serve-$name: server exited after $(( $(date +%s) - t0 )) s" >&2
      tail -25 "$log" >&2 || true
      SERVER_PID=""
      return 1
    fi
    code=$(curl -s -m 5 -o "$W/server-$name-health.body" -w "%{http_code}" "http://127.0.0.1:$PORT/health" || true)
    if [ "$code" = 200 ]; then
      echo "serve-$name: healthy after $(( $(date +%s) - t0 )) s"
      ready=1
      break
    fi
  done
  if [ "$ready" -ne 1 ]; then
    echo "serve-$name: not healthy after 480 s" >&2
    tail -25 "$log" >&2 || true
    cleanup
    SERVER_PID=""
    return 1
  fi

  # Check startup line reports the caches fit >= 262,144 tokens (fatal)
  "$UV" run python - "$log" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"the caches fit (\d+)", text)
assert m, f"startup line missing 'the caches fit N' in {sys.argv[1]}"
fits = int(m.group(1))
print(f"startup caches fit: {fits}")
assert fits >= 262144, f"caches fit {fits} < 262144"
PY

  "$UV" run python -m ink2q.tfrouter turns \
    --base-url "http://127.0.0.1:$PORT/v1" --model inkling2 \
    --messages "$TURNS_MSGS" --out "$out_json" 2>&1 | tee "$W/turns-$name.log"

  if [ "$name" = "cache-on" ] && command -v footprint >/dev/null 2>&1; then
    footprint "$SERVER_PID" > "$W/footprint-cache-on.txt" 2>&1 || true
  fi

  cleanup
  SERVER_PID=""
}

serve_turns cache-on
serve_turns cache-off --prompt-cache-gib 0

must turns-same "$UV" run python -m ink2q.tfrouter same --a "$W/turns-cache-on.json" --b "$W/turns-cache-off.json"

# Check cached_tokens > 0 on follow/regenerate/edit and 32k reask (fatal)
must turns-cached-check "$UV" run python - "$W/turns-cache-on.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for conv in d["conversations"]:
    label = conv["label"]
    for i, t in enumerate(conv["turns"]):
        op = t["op"]
        ct = t.get("cached_tokens", 0)
        if op in ("follow", "regenerate", "edit"):
            assert ct > 0, f"{label} turn {i} ({op}): expected cached_tokens > 0, got {ct}"
        elif op == "reask" and "32k" in label:
            assert ct > 0, f"{label} turn {i} (reask): expected cached_tokens > 0, got {ct}"
print("turns cached_tokens check: pass")
PY

# 7. diagnostic summary (non-fatal)
leg diag-summary "$UV" run python - "$W/turns-cache-on.json" "$W/turns-cache-off.json" "$W/server-cache-on.log" <<'PY'
import json, sys
on = json.load(open(sys.argv[1]))
off = json.load(open(sys.argv[2]))
for c_on, c_off in zip(on["conversations"], off["conversations"]):
    print(f"=== {c_on['label']} ===")
    for i, (t1, t0) in enumerate(zip(c_on["turns"], c_off["turns"])):
        r1, r0 = t1.get("decode_tok_s"), t0.get("decode_tok_s")
        ratio = (r1 / r0) if (r1 and r0) else None
        print(json.dumps({
            "turn": i, "op": t1["op"],
            "ttft_on": round(t1["ttft_s"], 3), "ttft_off": round(t0["ttft_s"], 3),
            "cached": t1["cached_tokens"], "prompt": t1["prompt_tokens"],
            "decode_on": round(r1, 2) if r1 else None, "decode_off": round(r0, 2) if r0 else None,
            "decode_ratio": round(ratio, 3) if ratio else None,
        }))
print("=== startup lines ===")
for line in open(sys.argv[3]):
    if "inkling2:" in line:
        print(line.rstrip())
PY

if [ -n "$FAILED" ]; then
  echo "cache gate completed with diagnostic failures:$FAILED"
fi
echo "CACHE_GATE_PASS"
