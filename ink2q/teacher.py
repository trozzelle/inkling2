"""Stream the MXFP8 original over a token set, one layer at a time; save each sequence's final hidden state."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import Floor, apply_limits
from ink2q.source import SourceIndex
from ink2q.tokens import load_tokens


def run(src: SourceIndex, seqs: list[np.ndarray], out: Path, floor: Floor | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    m = M.build(src.config())
    M.load_globals(m, src)
    states = [S.embed(m, mx.array(s)) for s in seqs]
    mx.eval([st.h for st in states])
    tokens = int(sum(len(s) for s in seqs))
    with open(out / "layers.jsonl", "w") as log:
        for layer in range(m.args.text.num_hidden_layers):
            t0 = time.monotonic()
            M.load_layer(m, src, layer)
            mx.eval(m.model.language_model.layers[layer].parameters())
            t1 = time.monotonic()
            for i, st in enumerate(states):
                states[i] = S.step(m, layer, st)
                mx.eval(states[i].h, states[i].blocks)
            M.drop_layer(m, layer)
            mx.clear_cache()
            rec = {"layer": layer, "seconds": round(time.monotonic() - t0, 2),
                   "load_seconds": round(t1 - t0, 2), "tokens": tokens}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if floor is not None:
                floor.check(f"layer {layer}")
    for i, st in enumerate(states):
        final = S.final_hidden(m, st)
        mx.save_safetensors(str(out / f"seq-{i:05d}.safetensors"), {"final": final})
    (out / "done.json").write_text(json.dumps({"sequences": len(seqs), "tokens": tokens}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--memory-gib", type=float, default=150)
    ap.add_argument("--cache-gib", type=float, default=4)
    ap.add_argument("--floor-gib", type=float, default=24)
    a = ap.parse_args()
    apply_limits(a.memory_gib, a.cache_gib)
    seqs = load_tokens(a.tokens)
    if a.limit:
        seqs = seqs[: a.limit]
    run(SourceIndex(a.source), seqs, a.out, Floor(a.floor_gib))


if __name__ == "__main__":
    main()
