"""MLX affine code packing: an LSB-first bitstream per row, value i at bit i*bits, in uint32 words.
pack/unpack are copied from the calq project; pack_affine verifies the result against mx.dequantize."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

BITS = (2, 3, 4, 5, 6, 8)
_CHUNK_ELEMS = 1 << 20


def pack(codes: np.ndarray, bits: int) -> np.ndarray:
    if bits not in BITS:
        raise ValueError(f"bits {bits} not in {BITS}")
    codes = np.asarray(codes)
    k = codes.shape[-1]
    if (k * bits) % 32:
        raise ValueError(f"row length {k} x {bits} bits is not a whole number of words")
    if codes.size and int(codes.min()) < 0:
        raise ValueError(f"code {int(codes.min())} is negative")
    if codes.size and int(codes.max()) >= 1 << bits:
        raise ValueError(f"code {int(codes.max())} does not fit {bits} bits")
    flat = codes.reshape(-1, k)
    out = np.empty((flat.shape[0], k * bits // 32), dtype=np.uint32)
    sb, s32 = np.arange(bits, dtype=np.uint32), np.arange(32, dtype=np.uint32)
    rows = max(1, _CHUNK_ELEMS // k)
    for a in range(0, flat.shape[0], rows):
        c = flat[a:a + rows].astype(np.uint32)
        b = ((c[..., None] >> sb) & 1).reshape(c.shape[0], -1, 32)
        out[a:a + rows] = (b << s32).sum(-1, dtype=np.uint32)
    return out.reshape(*codes.shape[:-1], k * bits // 32)


def unpack(words: np.ndarray, bits: int, k: int) -> np.ndarray:
    if bits not in BITS:
        raise ValueError(f"bits {bits} not in {BITS}")
    words = np.asarray(words, dtype=np.uint32)
    if words.shape[-1] * 32 != k * bits:
        raise ValueError(f"{words.shape[-1]} words per row do not hold {k} x {bits}-bit codes")
    flat = words.reshape(-1, words.shape[-1])
    out = np.empty((flat.shape[0], k), dtype=np.uint8)
    s32, sb = np.arange(32, dtype=np.uint32), np.arange(bits, dtype=np.uint32)
    rows = max(1, _CHUNK_ELEMS // k)
    for a in range(0, flat.shape[0], rows):
        b = (flat[a:a + rows][..., None] >> s32) & 1
        b = b.reshape(b.shape[0], k, bits)
        out[a:a + rows] = (b << sb).sum(-1, dtype=np.uint32).astype(np.uint8)
    return out.reshape(*words.shape[:-1], k)


def pack_affine(codes, scales, biases, bits, group):
    """Codes [...,N,K] uint8 and per-group scales/biases -> (uint32 words, bf16 scales, bf16 biases), checked so
    that mx.dequantize reproduces bf16(scale*q + bias) bit for bit."""
    codes = np.asarray(codes, np.uint8)
    words = pack(codes, bits)
    if not np.array_equal(unpack(words, bits, codes.shape[-1]), codes):
        raise ValueError("packed words do not unpack to the codes")
    s = mx.array(np.asarray(scales, np.float32)).astype(mx.bfloat16)
    b = mx.array(np.asarray(biases, np.float32)).astype(mx.bfloat16)
    w = mx.array(words)
    got = mx.dequantize(w, s, b, group_size=group, bits=bits)
    lead, K = codes.shape[:-1], codes.shape[-1]
    want = (mx.array(codes).astype(mx.float32).reshape(*lead, K // group, group) * s.astype(mx.float32)[..., None]
            + b.astype(mx.float32)[..., None]).reshape(*lead, K).astype(mx.bfloat16)
    if not bool(mx.array_equal(got.view(mx.uint16), want.view(mx.uint16))):
        raise ValueError("mx.dequantize of the packed words differs from bf16(scale*q + bias)")
    return w, s, b
