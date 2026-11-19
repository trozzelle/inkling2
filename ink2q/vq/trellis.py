"""QTIP 3INST-style computed trellis decode (timing study). k bits per weight; weight i uses the 16-bit window at
bit k*i of the row's tail-biting bitstream; value = sum of the two fp16 halves of ((a*v + b) & 0x8FFF8FFF) ^ 0x3B603B60."""
import mlx.core as mx
import numpy as np

A, B, MASK, XOR = 89226354, 64248484, 0x8FFF8FFF, 0x3B603B60


def random_payload(rng, E, N, K, k):
    return {"bits": mx.array(rng.integers(0, 2**32, size=(E, N, k * K // 32), dtype=np.uint32)),
            "scale": mx.array((rng.random((E, N)) * 0.01).astype(np.float16))}


def decode_ref(p, e, N, K, k):
    words = np.array(p["bits"][e]).astype(np.uint64)
    nw = words.shape[1]
    out = np.empty((N, K), np.float32)
    for i in range(K):
        bit = k * i
        w0, w1 = words[:, (bit >> 5) % nw], words[:, ((bit >> 5) + 1) % nw]
        v = ((w0 | (w1 << np.uint64(32))) >> np.uint64(bit & 31)) & np.uint64(0xFFFF)
        x = (np.uint64(A) * v + np.uint64(B)) & np.uint64(0xFFFFFFFF)
        m = ((x & np.uint64(MASK)) ^ np.uint64(XOR)).astype(np.uint32)
        h = m.view(np.float16).reshape(N, 2).astype(np.float32)
        out[:, i] = h.sum(1)
    return out * np.array(p["scale"][e]).astype(np.float32)[:, None]


_SRC = """
    threadgroup float xs[K];
    uint tid = thread_position_in_threadgroup.x, ts = thread_position_in_grid.y, t = ts / S;
    for (uint i = tid; i < uint(K); i += 256u) xs[i] = float(x[t * K + i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint NW = KB * K / 32;
    uint lane = tid % 32u, row0 = (thread_position_in_grid.x / 32u) * RPS;
    for (uint r = 0; r < uint(RPS); ++r) {
        ulong base = ulong(idx[ts]) * N + row0 + r;
        const device uint* w = bits + base * NW;
        float acc = 0.0f;
        for (uint i0 = lane * 8u; i0 < uint(K); i0 += 256u) {
            for (uint j = 0; j < 8u; ++j) {
                uint bit = KB * (i0 + j), wi = bit >> 5;
                ulong win = (ulong(w[(wi + 1u) % NW]) << 32) | ulong(w[wi % NW]);
                uint v = uint(win >> (bit & 31u)) & 0xFFFFu;
                uint m = ((89226354u * v + 64248484u) & 0x8FFF8FFFu) ^ 0x3B603B60u;
                half2 h = as_type<half2>(m);
                acc += (float(h.x) + float(h.y)) * xs[i0 + j];
            }
        }
        acc = simd_sum(acc);
        if (lane == 0u) y[ulong(ts) * N + row0 + r] = acc * float(scale[base]);
    }
"""
_k = mx.fast.metal_kernel(name="ink2q_trellis_gemv", input_names=["x", "idx", "bits", "scale"],
                          output_names=["y"], source=_SRC)


def gemv(x, idx, p, N, K, k, rps=4):
    T, S = x.shape[0], idx.shape[1]
    (y,) = _k(inputs=[x, idx, p["bits"], p["scale"]], template=[("K", K), ("N", N), ("S", S), ("RPS", rps), ("KB", k)],
              grid=(N // rps * 32, T * S, 1), threadgroup=(256, 1, 1), output_shapes=[(T, S, N)],
              output_dtypes=[mx.float32])
    return y
