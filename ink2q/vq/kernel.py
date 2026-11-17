"""Metal kernels for R13+8 expert codes (layout: ink2q.vq.format).

Decode GEMV: one simdgroup per output row, RPS rows per simdgroup, lanes strided over 8-weight groups.
The stage-1 entry is gathered as its raw 8 E4M3 bytes and widened to fp16 with bit operations
(byte b -> half bits ((b & 0x80) << 8) | ((b & 0x7F) << 7), which equals e4m3(b) * 2^-8 exactly,
subnormals included); the factor 2^8 is applied once with the row gain. Each uint holds bytes
(0, 1, 2, 3) and widens to the order (0, 2, 1, 3), so x and the stage-2 table are stored in that
order. Stage-2 entries live in threadgroup memory as fp16 pre-scaled by 2^-8; both stages are
widened to fp32 before they are added, so the sum is exact.
"""
from typing import NamedTuple

import mlx.core as mx
import numpy as np

from ink2q.vq.format import e4m3_table

RPS = 4  # rows per simdgroup (measured best on the M5 Max, 2026-11-17)
_PERM = [0, 2, 1, 3, 4, 6, 5, 7]


class VQWeights(NamedTuple):
    lo12: mx.array
    hi1: mx.array
    i2: mx.array
    gain: mx.array
    c1: mx.array
    c2: mx.array


_HEADER = """
inline float e4m3_to_float(uint b) {
    uint e = (b >> 3) & 15u;
    uint m = b & 7u;
    float v = e == 0u ? float(m) * 0.001953125f : (1.0f + float(m) * 0.125f) * exp2(float(int(e) - 7));
    return (b & 128u) != 0u ? -v : v;
}
inline uint stage1_index(const device uint8_t* lo, const device uint8_t* hi, uint g) {
    uint p = (g >> 1) * 3u;
    uint b0 = lo[p], b1 = lo[p + 1], b2 = lo[p + 2];
    uint low = (g & 1u) == 0u ? (b0 | ((b1 & 15u) << 8)) : ((b1 >> 4) | (b2 << 4));
    return low | (((uint(hi[g >> 3]) >> (g & 7u)) & 1u) << 12);
}
"""

_GEMV = """
    threadgroup half4 c2s[512];
    threadgroup float4 xs[K / 4];
    uint tid = thread_position_in_threadgroup.x;
    uint ts = thread_position_in_grid.y;
    uint t = ts / S;
    const device half4* c2v = (const device half4*)c2p;
    for (uint i = tid; i < 512u; i += 256u) c2s[i] = c2v[i];
    const device bfloat16_t* xr = x + t * K;
    for (uint i = tid; i < uint(K) / 4u; i += 256u)
        xs[i] = float4(float(xr[4*i]), float(xr[4*i+2]), float(xr[4*i+1]), float(xr[4*i+3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint G = K / 8;
    uint lane = tid % 32u;
    uint row0 = (thread_position_in_grid.x / 32u) * RPS;
    ulong ebase = ulong(idx[ts]) * N;
    const device uint2* c1v = (const device uint2*)c1;
    for (uint r = 0; r < uint(RPS); ++r) {
        ulong base = ebase + row0 + r;
        const device uint8_t* lo = lo12 + base * (G * 3 / 2);
        const device uint8_t* hi = hi1 + base * (G / 8);
        const device uint8_t* j2 = i2 + base * G;
        float acc = 0.0f;
        for (uint g = lane; g < G; g += 32u) {
            uint i1 = stage1_index(lo, hi, g);
            uint k2 = uint(j2[g]) * 2u;
            uint2 w = c1v[i1];
            half2 a0 = as_type<half2>(((w.x & 0x00800080u) << 8) | ((w.x & 0x007F007Fu) << 7));
            half2 b0 = as_type<half2>((w.x & 0x80008000u) | ((w.x & 0x7F007F00u) >> 1));
            half2 a1 = as_type<half2>(((w.y & 0x00800080u) << 8) | ((w.y & 0x007F007Fu) << 7));
            half2 b1 = as_type<half2>((w.y & 0x80008000u) | ((w.y & 0x7F007F00u) >> 1));
            float4 v0 = float4(half4(a0, b0)) + float4(c2s[k2]);
            float4 v1 = float4(half4(a1, b1)) + float4(c2s[k2 + 1u]);
            acc += dot(v0, xs[2u * g]) + dot(v1, xs[2u * g + 1u]);
        }
        acc = simd_sum(acc);
        if (lane == 0u) y[ulong(ts) * N + row0 + r] = acc * (256.0f * float(gain[base]));
    }
"""

_DEQUANT = """
    threadgroup float lut[256];
    uint tid = thread_position_in_threadgroup.x;
    for (uint i = tid; i < 256u; i += 32u) lut[i] = e4m3_to_float(i);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint G = K / 8;
    uint g = thread_position_in_grid.x, row = thread_position_in_grid.y, m = thread_position_in_grid.z;
    ulong base = ulong(experts[m]) * N + row;
    uint i1 = stage1_index(lo12 + base * (G * 3 / 2), hi1 + base * (G / 8), g);
    uint k2 = uint(i2[base * G + g]) * 8u;
    uint2 w = ((const device uint2*)c1)[i1];
    float gn = float(gain[base]);
    device bfloat16_t* o = out + (ulong(m) * N + row) * K + g * 8u;
    for (uint d = 0; d < 4u; ++d) {
        o[d] = bfloat16_t((lut[(w.x >> (8u * d)) & 255u] + lut[c2[k2 + d]]) * gn);
        o[4u + d] = bfloat16_t((lut[(w.y >> (8u * d)) & 255u] + lut[c2[k2 + 4u + d]]) * gn);
    }
"""

_gemv = mx.fast.metal_kernel(
    name="ink2q_vq_r13p8_gemv",
    input_names=["x", "idx", "lo12", "hi1", "i2", "gain", "c1", "c2p"],
    output_names=["y"],
    source=_GEMV,
    header=_HEADER,
)
_dequant = mx.fast.metal_kernel(
    name="ink2q_vq_r13p8_dequant",
    input_names=["experts", "lo12", "hi1", "i2", "gain", "c1", "c2"],
    output_names=["out"],
    source=_DEQUANT,
    header=_HEADER,
)


def stage2_table(c2: mx.array) -> mx.array:
    """The GEMV's stage-2 table: fp16 [256*8], values * 2^-8, each 8-group in the order (0,2,1,3,4,6,5,7)."""
    v = e4m3_table()[np.array(c2).reshape(256, 8)] / 256.0
    return mx.array(v[:, _PERM].astype(np.float16).reshape(-1))


def gather_gemv(x: mx.array, idx: mx.array, w: VQWeights, c2p: mx.array | None = None) -> mx.array:
    T, K = x.shape
    S = idx.shape[1]
    N = w.gain.shape[1]
    assert N % (8 * RPS) == 0 and K % 128 == 0
    if c2p is None:
        c2p = stage2_table(w.c2)
    (y,) = _gemv(
        inputs=[x, idx.astype(mx.uint32), w.lo12, w.hi1, w.i2, w.gain, w.c1, c2p],
        template=[("K", K), ("N", N), ("S", S), ("RPS", RPS)],
        grid=(N // RPS * 32, T * S, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(T, S, N)],
        output_dtypes=[mx.float32],
    )
    return y


def dequant(experts: mx.array, w: VQWeights) -> mx.array:
    N = w.gain.shape[1]
    K = w.i2.shape[2] * 8
    M = experts.shape[0]
    (out,) = _dequant(
        inputs=[experts.astype(mx.uint32), w.lo12, w.hi1, w.i2, w.gain, w.c1, w.c2],
        template=[("K", K), ("N", N)],
        grid=(K // 8, N, M),
        threadgroup=(32, 1, 1),
        output_shapes=[(M, N, K)],
        output_dtypes=[mx.bfloat16],
    )
    return out
