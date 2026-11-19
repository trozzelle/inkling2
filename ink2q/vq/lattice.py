"""IQ3_XXS-style sign-lattice decode (timing study; random grid). Per 256 weights: d f16, 64 grid bytes, 8 aux u32."""
import mlx.core as mx
import numpy as np

KSIGNS = np.array([i | ((bin(i).count("1") & 1) << 7) for i in range(128)], np.uint8)


def random_payload(rng, E, N, K):
    return {"d": mx.array((rng.random((E, N, K // 256)) * 0.01).astype(np.float16)),
            "qs": mx.array(rng.integers(0, 256, size=(E, N, K // 4), dtype=np.uint8)),
            "aux": mx.array(rng.integers(0, 2**32, size=(E, N, K // 32), dtype=np.uint32)),
            "grid": mx.array(rng.integers(0, 2**32, size=(256,), dtype=np.uint32)),
            "ks": mx.array(KSIGNS)}


def decode_ref(p, e, N, K):
    d = np.array(p["d"][e]).astype(np.float32); qs = np.array(p["qs"][e]); aux = np.array(p["aux"][e])
    grid = np.array(p["grid"]).view(np.uint8).reshape(256, 4).astype(np.float32)
    out = np.empty((N, K), np.float32)
    for sb in range(K // 32):
        a = aux[:, sb]
        db = d[:, sb // 8] * (0.5 + (a >> 28)) * 0.5
        for l in range(4):
            s = KSIGNS[(a >> (7 * l)) & 127]
            mags = np.concatenate([grid[qs[:, sb * 8 + 2 * l]], grid[qs[:, sb * 8 + 2 * l + 1]]], axis=1)
            sign = np.where((s[:, None] >> np.arange(8)) & 1, -1.0, 1.0)
            out[:, sb * 32 + 8 * l: sb * 32 + 8 * l + 8] = db[:, None] * mags * sign
    return out


_SRC = """
    threadgroup uint gs[256];
    threadgroup uchar kss[128];
    threadgroup float xs[K];
    uint tid = thread_position_in_threadgroup.x, ts = thread_position_in_grid.y, t = ts / S;
    for (uint i = tid; i < 256u; i += 256u) gs[i] = grid[i];
    for (uint i = tid; i < 128u; i += 256u) kss[i] = ks[i];
    for (uint i = tid; i < uint(K); i += 256u) xs[i] = float(x[t * K + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint lane = tid % 32u, row0 = (thread_position_in_grid.x / 32u) * RPS;
    for (uint r = 0; r < uint(RPS); ++r) {
        ulong base = ulong(idx[ts]) * N + row0 + r;
        float acc = 0.0f;
        for (uint sb = lane; sb < uint(K) / 32u; sb += 32u) {
            uint a = aux[base * (K / 32) + sb];
            float db = float(d[base * (K / 256) + sb / 8u]) * (0.5f + float(a >> 28)) * 0.5f;
            const device uchar* q = qs + base * (K / 4) + sb * 8u;
            float s8 = 0.0f;
            for (uint l = 0; l < 4u; ++l) {
                uint sg = kss[(a >> (7u * l)) & 127u];
                uint g1 = gs[q[2u * l]], g2 = gs[q[2u * l + 1u]];
                uint o = sb * 32u + 8u * l;
                for (uint j = 0; j < 4u; ++j) {
                    s8 += float((g1 >> (8u * j)) & 255u) * ((sg >> j) & 1u ? -1.0f : 1.0f) * xs[o + j];
                    s8 += float((g2 >> (8u * j)) & 255u) * ((sg >> (j + 4u)) & 1u ? -1.0f : 1.0f) * xs[o + 4u + j];
                }
            }
            acc += db * s8;
        }
        acc = simd_sum(acc);
        if (lane == 0u) y[ulong(ts) * N + row0 + r] = acc;
    }
"""
_k = mx.fast.metal_kernel(name="ink2q_lattice_gemv", input_names=["x", "idx", "d", "qs", "aux", "grid", "ks"],
                          output_names=["y"], source=_SRC)


def gemv(x, idx, p, N, K, rps=4):
    T, S = x.shape[0], idx.shape[1]
    (y,) = _k(inputs=[x, idx, p["d"], p["qs"], p["aux"], p["grid"], p["ks"]],
              template=[("K", K), ("N", N), ("S", S), ("RPS", rps)], grid=(N // rps * 32, T * S, 1),
              threadgroup=(256, 1, 1), output_shapes=[(T, S, N)], output_dtypes=[mx.float32])
    return y
