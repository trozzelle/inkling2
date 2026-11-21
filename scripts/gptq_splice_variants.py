"""Full-layer statistics on the 1.0M-token calibration subset, FP64 GPTQ at 2/3/4 bits on both projections of
three chosen layers, candidate directories and the splice variants (GPTQ, RTN and oQe controls)."""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q import collect as C
from ink2q import model as M
from ink2q import stream as S
from ink2q.assemble import write_expert_index
from ink2q.memguard import Floor, apply_limits
from ink2q.quant.layer import convert_layer
from ink2q.source import SourceIndex
from ink2q.tokens import load_tokens

BITS = [2, 3, 4]
CANDIDATES = [(2, 2), (3, 3), (4, 4), (2, 3), (3, 2)]       # (w13 bits, w2 bits); the splits measure the w13 share


def variants(layers, out: Path, oqe: str) -> list[dict]:
    vs = []
    for l in layers:
        for b13, b2 in CANDIDATES:
            vs.append({"name": f"gptq_{b13}_{b2}_L{l}",
                       "layers": {str(l): {"kind": "mlx", "dir": str(out / "cand" / f"L{l:02d}_{b13}_{b2}")}}})
        for b in BITS:
            vs.append({"name": f"rtn{b}_L{l}", "layers": {str(l): {"kind": "rtn", "bits_gu": b, "bits_dn": b}}})
        vs.append({"name": f"oqe_L{l}", "layers": {str(l): {"kind": "mlx", "dir": oqe}}})
    # whole-model floor: an 8-bit perturbation of every layer bounds what any quantization can reach; 4-bit and
    # all-oQe place the headroom between that floor and the baseline
    for b in (8, 4):
        vs.append({"name": f"rtn{b}_all", "layers": {str(l): {"kind": "rtn", "bits_gu": b, "bits_dn": b} for l in range(48)}})
    vs.append({"name": "oqe_all", "layers": {str(l): {"kind": "mlx", "dir": oqe} for l in range(48)}})
    return vs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--calib", type=Path, required=True)
    ap.add_argument("--subset", type=Path, required=True, help="e2-calib-subset.json")
    ap.add_argument("--e2", type=Path, required=True)
    ap.add_argument("--oqe", required=True)
    ap.add_argument("--layers", required=True, help="comma-separated layer indices")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--variants", type=Path, required=True)
    ap.add_argument("--memory-gib", type=float, default=170)
    a = ap.parse_args()
    apply_limits(a.memory_gib, 4)
    floor = Floor(24)
    layers = sorted(int(x) for x in a.layers.split(","))
    cfg = {**json.loads(a.e2.read_text())["best"], "group": 128, "batch": 8}
    cal = load_tokens(a.calib)
    seqs = [cal[i] for i in json.loads(a.subset.read_text())["indices"]]
    a.out.mkdir(parents=True, exist_ok=True)
    src = SourceIndex(a.source)
    m = M.build(src.config())
    M.load_globals(m, src, final=False)
    states = [S.embed_lean(m, mx.array(s)) for s in seqs]
    mx.eval([st.h for st in states])
    receipts = []
    for layer in range(max(layers) + 1):
        t0 = time.monotonic()
        M.load_layer(m, src, layer)
        mod = m.model.language_model.layers[layer]
        if layer in layers:
            st = C.ExpertStats(mod, None, raw=cfg["kind"] == "raw", normalized=cfg["kind"] == "norm")
            states = C.step_with_stats(m, layer, states, st)
            t_stats = time.monotonic() - t0
            host = st.to_host(np.float32, release=True)
            del st
            avail = floor.check(f"layer {layer} statistics")
            rec = convert_layer(mod, layer, host, {"w13": BITS, "w2": BITS}, cfg, a.out, floor=floor)
            del host
            rec.update(stats_seconds=round(t_stats, 1), available_gib=round(avail / 2**30, 1))
            receipts.append(rec)
            print(json.dumps(rec), flush=True)
            for b13, b2 in CANDIDATES:
                write_expert_index({str(layer): {"w13": b13, "w2": b2}}, a.out, a.out / "cand" / f"L{layer:02d}_{b13}_{b2}")
        else:
            states = [S.step_lean(m, layer, s) for s in states]
            mx.eval([s.h for s in states])
        M.drop_layer(m, layer)
        mx.clear_cache()
        floor.check(f"layer {layer}")
        print(json.dumps({"layer": layer, "seconds": round(time.monotonic() - t0, 1)}), flush=True)
    (a.out / "receipts.json").write_text(json.dumps(receipts, indent=1) + "\n")
    a.variants.write_text(json.dumps(variants(layers, a.out, a.oqe), indent=1) + "\n")


if __name__ == "__main__":
    main()
