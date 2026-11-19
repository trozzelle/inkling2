"""E1: does affine gather_qmm time track bytes? Chained calls at bits 2/3/4, T in {1, 8}, real expert shapes."""
import json
import sys

import mlx.core as mx
import numpy as np

from ink2q.memguard import apply_limits
from ink2q.vq.bench import E, LAYERS, S, timed

SHAPES = {"w13": (4608, 4608), "w2": (4608, 2304), "w2_pad2560": (4608, 2560)}


def weights(rng, N, K, bits):
    """Random packed affine g128 weights: timing does not depend on values."""
    wq = mx.array(rng.integers(0, 2**32, size=(E, N, K * bits // 32), dtype=np.uint32))
    sc = mx.array((rng.random((E, N, K // 128)) * 0.01).astype(np.float32)).astype(mx.bfloat16)
    bi = mx.array((rng.random((E, N, K // 128)) * 0.01).astype(np.float32)).astype(mx.bfloat16)
    return wq, sc, bi


def main(out):
    apply_limits(16, 2)
    rng = np.random.default_rng(0)
    rows = []
    for name, (N, K) in SHAPES.items():
        for bits in (2, 3, 4):
            wq, sc, bi = weights(rng, N, K, bits)
            mx.eval(wq, sc, bi)
            for T in (1, 8):
                x = mx.random.normal((T, K)).astype(mx.bfloat16)
                ids_np = [rng.integers(0, E, size=(T, S)).astype(np.uint32) for _ in range(LAYERS)]
                ids = [mx.array(i) for i in ids_np]
                mx.eval(x, ids)

                def f():
                    y = x
                    for i in ids:
                        o = mx.gather_qmm(mx.expand_dims(y, (-2, -3)), wq, sc, bi, rhs_indices=i, transpose=True,
                                          group_size=128, bits=bits)
                        y = o[:, 0, 0, :K].astype(mx.bfloat16) * 0 + x
                    return y

                ms = timed(f, iters=10, warmup=3) / LAYERS * 1e3
                experts = float(np.mean([len(np.unique(i)) for i in ids_np]))   # distinct experts read per call
                byt = experts * N * K * bits / 8 * (1 + 32 / (bits * 128))
                rows.append({"shape": name, "bits": bits, "T": T, "ms": ms, "experts_per_call": experts,
                             "bytes": byt, "gb_s": byt / ms / 1e6})
                print(json.dumps(rows[-1]), flush=True)
            del wq, sc, bi
            mx.clear_cache()
    with open(out, "w") as fh:
        json.dump(rows, fh, indent=1)
    for name in SHAPES:
        for T in (1, 8):
            t = {r["bits"]: r["ms"] for r in rows if r["shape"] == name and r["T"] == T}
            print(f"{name} T={T}: t2/t3={t[2] / t[3]:.3f} (bytes 0.69)  t4/t3={t[4] / t[3]:.3f} (bytes 1.31)")


if __name__ == "__main__":
    main(sys.argv[1])
