"""Diagnostic: stream the 1.0M-token calibration subset to one layer and save raw Grams for a few experts plus the
all-token Gram (from which the w13 prior is built), so GPTQ on that layer can be replayed offline."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q import collect as C
from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import Floor, apply_limits
from ink2q.source import SourceIndex
from ink2q.tokens import load_tokens


class AllToken(C.ExpertStats):
    """ExpertStats plus the Gram of every token (the layer-mean prior is Σ_e S_e / Σ_e n_e = 8·Σxxᵀ / (8·T))."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.all = None
        self.tokens = 0

    def add(self, x):
        x2 = x.reshape(-1, x.shape[-1])
        xf = x2.astype(mx.float32)
        g = xf.T @ xf
        self.all = g if self.all is None else self.all + g
        mx.eval(self.all)
        self.tokens += x2.shape[0]
        super().add(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--experts", required=True, help="first-last")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    apply_limits(150, 4)
    floor = Floor(24)
    lo, hi = map(int, a.experts.split("-"))
    cal = load_tokens(a.calib)
    seqs = [cal[i] for i in json.loads(a.subset.read_text())["indices"]]
    src = SourceIndex(a.source)
    m = M.build(src.config())
    M.load_globals(m, src, final=False)
    states = [S.embed_lean(m, mx.array(s)) for s in seqs]
    for layer in range(a.layer + 1):
        M.load_layer(m, src, layer)
        if layer == a.layer:
            st = AllToken(m.model.language_model.layers[layer], list(range(lo, hi + 1)), raw=True, normalized=False)
            C.step_with_stats(m, layer, states, st)
            host = st.to_host(np.float32, release=True)
            a.out.mkdir(parents=True, exist_ok=True)
            np.savez(a.out / f"L{layer}_e{lo}-{hi}.npz", all=np.array(st.all, np.float64), tokens=st.tokens, **host)
        else:
            states = [S.step_lean(m, layer, s) for s in states]
            mx.eval([s.h for s in states])
        M.drop_layer(m, layer)
        mx.clear_cache()
        floor.check(f"layer {layer}")
        print(json.dumps({"layer": layer}), flush=True)


if __name__ == "__main__":
    main()
