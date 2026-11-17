"""G0: R13+8 decode GEMV against affine 3-bit g128 gather_qmm at the real Inkling 2 expert shapes."""
import argparse
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx
import numpy as np

from ink2q.vq.kernel import VQWeights, dequant, gather_gemv, stage2_table

E, S = 64, 8                                           # experts resident in the bench, slots per token
LAYERS = 48                                            # dependent calls per timed eval
SHAPES = {"w13": (4608, 4608), "w2": (4608, 2304)}     # (N out rows, K in cols)


def timed(fn, iters=100, warmup=10) -> float:
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def vq_weights(rng, N, K) -> VQWeights:
    G = K // 8
    c = rng.integers(0, 256, size=(8192 * 8 + 256 * 8,), dtype=np.uint8)
    c[(c & 127) == 127] = 0
    return VQWeights(
        lo12=mx.array(rng.integers(0, 256, size=(E, N, G * 3 // 2), dtype=np.uint8)),
        hi1=mx.array(rng.integers(0, 256, size=(E, N, G // 8), dtype=np.uint8)),
        i2=mx.array(rng.integers(0, 256, size=(E, N, G), dtype=np.uint8)),
        gain=mx.array((rng.random((E, N)) * 0.01).astype(np.float16)),
        c1=mx.array(c[: 8192 * 8]),
        c2=mx.array(c[8192 * 8 :]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for name, (N, K) in SHAPES.items():
        w = vq_weights(rng, N, K)
        dense = (mx.random.normal((E, N, K)) * 0.02).astype(mx.bfloat16)
        wq, sc, bi = mx.quantize(dense, group_size=128, bits=3)
        del dense
        c2p = stage2_table(w.c2)
        mx.eval(w, wq, sc, bi, c2p)
        for T in (1, 8):
            x = mx.random.normal((T, K)).astype(mx.bfloat16)
            ids = [mx.array(rng.integers(0, E, size=(T, S)).astype(np.uint32)) for _ in range(LAYERS)]
            mx.eval(x, ids)
            # LAYERS dependent calls per eval, as one decode token through the model: per-eval sync
            # overhead (about 0.1-0.2 ms) is amortized instead of dominating sub-millisecond kernels.
            def vq():
                y = x
                for i in ids:
                    y = gather_gemv(y, i, w, c2p)[:, 0, :K].astype(mx.bfloat16) * 0 + x
                return y

            def affine():
                y = x
                for i in ids:
                    o = mx.gather_qmm(mx.expand_dims(y, (-2, -3)), wq, sc, bi, rhs_indices=i,
                                      transpose=True, group_size=128, bits=3)
                    y = o[:, 0, 0, :K].astype(mx.bfloat16) * 0 + x
                return y

            t_vq = timed(vq, iters=10, warmup=3) / LAYERS
            t_aff = timed(affine, iters=10, warmup=3) / LAYERS
            rows.append({"shape": name, "T": T, "vq_ms": t_vq * 1e3, "affine3_ms": t_aff * 1e3,
                         "ratio": t_vq / t_aff})
        ex = mx.array(rng.integers(0, E, size=(32,)).astype(np.uint32))
        t_dq = timed(lambda: dequant(ex, w), iters=20, warmup=3)
        rows.append({"shape": name, "dequant_32_experts_ms": t_dq * 1e3,
                     "dequant_gb_per_s_written": 32 * N * K * 2 / t_dq / 1e9})
    gate = all(r["ratio"] <= 1.25 for r in rows if "ratio" in r)
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    result = {"host": platform.node(), "chip": chip, "mlx": mx.__version__, "rows": rows, "G0_pass": gate}
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
