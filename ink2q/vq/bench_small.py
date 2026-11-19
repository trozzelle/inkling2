"""E3: small-table decode GEMVs (IQ3_XXS-style lattice, QTIP 3INST-style trellis) against affine 3-bit gather_qmm,
chained 48 calls per evaluation as in G0/E1, at the real expert shapes."""
import json
import platform
import subprocess
import sys

import mlx.core as mx
import numpy as np

from ink2q.memguard import apply_limits
from ink2q.vq import lattice, trellis
from ink2q.vq.bench import E, LAYERS, S, SHAPES, affine3_weights, timed


def _chain(x, ids, call, K):
    def f():
        y = x
        for i in ids:
            y = call(y, i)[:, 0, :K].astype(mx.bfloat16) * 0 + x
        return y
    return f


def main(out):
    apply_limits(16, 2)
    rng = np.random.default_rng(0)
    rows = []
    for name, (N, K) in SHAPES.items():
        wq, sc, bi = affine3_weights(rng, N, K)
        kinds = {"lattice": (lattice.random_payload(rng, E, N, K), {}, 3.0625)}
        for k in (2, 3):
            kinds[f"trellis{k}"] = (trellis.random_payload(rng, E, N, K, k), {"k": k}, k + 16 / K)
        mx.eval(wq, sc, bi, [p for p, _, _ in kinds.values()])
        for T in (1, 8):
            x = mx.random.normal((T, K)).astype(mx.bfloat16)
            ids = [mx.array(rng.integers(0, E, size=(T, S)).astype(np.uint32)) for _ in range(LAYERS)]
            mx.eval(x, ids)
            aff = lambda y, i: mx.gather_qmm(mx.expand_dims(y, (-2, -3)), wq, sc, bi, rhs_indices=i, transpose=True,
                                             group_size=128, bits=3)[:, :, 0, :]
            t_aff = timed(_chain(x, ids, aff, K), iters=20, warmup=3) / LAYERS
            for kind, (pay, kw, bpw) in kinds.items():
                mod = lattice if kind == "lattice" else trellis
                call = lambda y, i, mod=mod, pay=pay, kw=kw: mod.gemv(y, i, pay, N=N, K=K, **kw)
                t = timed(_chain(x, ids, call, K), iters=20, warmup=3) / LAYERS
                rows.append({"kind": kind, "bpw": bpw, "shape": name, "T": T, "ms": t * 1e3,
                             "affine3_ms": t_aff * 1e3, "ratio_per_weight": t / t_aff})
                print(json.dumps(rows[-1]), flush=True)
        del wq, sc, bi, kinds
        mx.clear_cache()
    passes = {}
    for kind in sorted({r["kind"] for r in rows}):
        passes[kind] = all(r["ratio_per_weight"] <= 1.15 for r in rows if r["kind"] == kind and r["T"] == 8)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    res = {"host": platform.node(), "chip": chip, "mlx": mx.__version__, "rows": rows, "E3_pass_T8": passes}
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(passes))


if __name__ == "__main__":
    main(sys.argv[1])
