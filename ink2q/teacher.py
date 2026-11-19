"""Stream the MXFP8 original (optionally with variant experts) over a token set, one layer at a time; save each
sequence's final hidden state."""
import argparse
import contextlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q import experts as X
from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import Floor, apply_limits
from ink2q.source import SourceIndex
from ink2q.tokens import load_tokens


def run(src: SourceIndex, seqs: list[np.ndarray], out: Path, floor: Floor | None = None, variant=None) -> None:
    """variant: callable (layer_module, layer) -> ExpertSpec | None; a returned spec replaces that layer's experts."""
    run_sets(src, [(seqs, out)], floor, variant)


def run_sets(src: SourceIndex, sets: list[tuple[list[np.ndarray], Path]], floor: Floor | None = None,
             variant=None) -> None:
    """One layer-streamed pass over several token sets; each set's finals go to its own directory."""
    outs = [Path(o) for _, o in sets]
    for o in outs:
        o.mkdir(parents=True, exist_ok=True)
    seqs = [s for ss, _ in sets for s in ss]
    m = M.build(src.config())
    M.load_globals(m, src)
    states = [S.embed_lean(m, mx.array(s)) for s in seqs]
    mx.eval([st.h for st in states])
    tokens = int(sum(len(s) for s in seqs))
    logs = [open(o / "layers.jsonl", "w") for o in outs]
    try:
        for layer in range(m.args.text.num_hidden_layers):
            t0 = time.monotonic()
            M.load_layer(m, src, layer)
            mod = m.model.language_model.layers[layer]
            mx.eval(mod.parameters())
            spec = variant(mod, layer) if variant else None
            label = spec.label if spec is not None else None
            t1 = time.monotonic()
            with X.install(mod, spec) if spec is not None else contextlib.nullcontext():
                for i, st in enumerate(states):
                    states[i] = S.step_lean(m, layer, st)
                    mx.eval(states[i].h)
            del spec
            M.drop_layer(m, layer)
            mx.clear_cache()
            rec = {"layer": layer, "seconds": round(time.monotonic() - t0, 2),
                   "load_seconds": round(t1 - t0, 2), "tokens": tokens, "variant": label}
            for log in logs:
                log.write(json.dumps(rec) + "\n")
                log.flush()
            print(json.dumps(rec), flush=True)
            if floor is not None:
                floor.check(f"layer {layer}")
    finally:
        for log in logs:
            log.close()
    k = 0
    for ss, o in sets:
        for i in range(len(ss)):
            final = S.final_hidden_lean(m, states[k])
            mx.save_safetensors(str(Path(o) / f"seq-{i:05d}.safetensors"), {"final": final})
            k += 1
        (Path(o) / "done.json").write_text(json.dumps({"sequences": len(ss), "tokens": int(sum(len(s) for s in ss)),
                                                       "variant": variant is not None}) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, nargs="+", required=True, help="one output directory per --tokens file")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--memory-gib", type=float, default=150)
    ap.add_argument("--cache-gib", type=float, default=4)
    ap.add_argument("--floor-gib", type=float, default=24)
    ap.add_argument("--variant-mlx", type=Path, help="oMLX-layout checkpoint whose experts replace every layer's")
    a = ap.parse_args()
    if len(a.tokens) != len(a.out):
        ap.error("--tokens and --out need the same number of paths")
    apply_limits(a.memory_gib, a.cache_gib)
    sets = []
    for t, o in zip(a.tokens, a.out):
        seqs = load_tokens(t)
        sets.append((seqs[: a.limit] if a.limit else seqs, o))
    variant = (lambda mod, l: X.from_mlx_dir(a.variant_mlx, l)) if a.variant_mlx else None
    run_sets(SourceIndex(a.source), sets, Floor(a.floor_gib), variant=variant)


if __name__ == "__main__":
    main()
