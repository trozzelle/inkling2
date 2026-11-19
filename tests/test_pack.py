import mlx.core as mx
import numpy as np
import pytest

from ink2q.quant.pack import pack_affine


@pytest.mark.parametrize("bits", [2, 3, 4])
def test_pack_matches_mx_dequantize(bits):
    rng = np.random.default_rng(bits)
    codes = rng.integers(0, 2**bits, size=(3, 8, 256), dtype=np.uint8)
    scales = rng.uniform(0.001, 0.02, size=(3, 8, 2)).astype(np.float32)
    biases = rng.uniform(-0.1, 0.0, size=(3, 8, 2)).astype(np.float32)
    words, s, b = pack_affine(codes, scales, biases, bits, group=128)
    assert words.shape == (3, 8, 256 * bits // 32) and words.dtype == mx.uint32
    got = mx.dequantize(words, s, b, group_size=128, bits=bits)
    want = (mx.array(codes).astype(mx.float32).reshape(3, 8, 2, 128) * s.astype(mx.float32)[..., None]
            + b.astype(mx.float32)[..., None]).reshape(3, 8, 256).astype(mx.bfloat16)
    assert mx.array_equal(got, want)


def test_pack_round_trips_mx_quantize():
    w = mx.random.normal((4, 256), key=mx.random.key(0)).astype(mx.bfloat16)
    for bits in (2, 3, 4):
        wq, s, b = mx.quantize(w, group_size=128, bits=bits)
        deq = mx.dequantize(wq, s, b, group_size=128, bits=bits)
        codes = np.rint((np.array(deq.astype(mx.float32)).reshape(4, 2, 128) - np.array(b.astype(mx.float32))[..., None])
                        / np.array(s.astype(mx.float32))[..., None]).reshape(4, 256).astype(np.uint8)
        words, _, _ = pack_affine(codes, np.array(s.astype(mx.float32)), np.array(b.astype(mx.float32)), bits, 128)
        assert mx.array_equal(words, wq)
