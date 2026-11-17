import mlx.core as mx
import numpy as np
import pytest

from ink2q.vq.format import decode
from ink2q.vq.kernel import VQWeights, dequant, gather_gemv


def random_weights(rng, E, N, K):
    G = K // 8
    c1 = rng.integers(0, 256, size=(8192, 8), dtype=np.uint8)
    c2 = rng.integers(0, 256, size=(256, 8), dtype=np.uint8)
    for c in (c1, c2):
        c[(c & 127) == 127] = 0  # no NaN codewords
    host = dict(
        lo12=rng.integers(0, 256, size=(E, N, G * 3 // 2), dtype=np.uint8),
        hi1=rng.integers(0, 256, size=(E, N, G // 8), dtype=np.uint8),
        i2=rng.integers(0, 256, size=(E, N, G), dtype=np.uint8),
        gain=(rng.random((E, N)) * 0.01).astype(np.float16),
        c1=c1,
        c2=c2,
    )
    dev = VQWeights(**{k: mx.array(v.reshape(-1) if k in ("c1", "c2") else v) for k, v in host.items()})
    return host, dev


@pytest.mark.parametrize("K", [256, 512])
def test_gather_gemv_matches_reference(K):
    rng = np.random.default_rng(1)
    E, N, T, S = 4, 64, 3, 2
    host, w = random_weights(rng, E, N, K)
    x = rng.standard_normal((T, K)).astype(np.float32)
    xb = mx.array(x).astype(mx.bfloat16)
    idx = rng.integers(0, E, size=(T, S)).astype(np.uint32)
    y = np.array(gather_gemv(xb, mx.array(idx), w))
    W = decode(host["lo12"], host["hi1"], host["i2"], host["gain"], host["c1"], host["c2"])  # [E,N,K]
    xr = np.array(xb.astype(mx.float32)).astype(np.float64)
    ref = np.einsum("tk,tsnk->tsn", xr, W[idx].astype(np.float64))
    np.testing.assert_allclose(y, ref, rtol=1e-3, atol=1e-3)


def test_dequant_matches_reference():
    rng = np.random.default_rng(2)
    host, w = random_weights(rng, 4, 32, 256)
    out = np.array(dequant(mx.array(np.array([2, 0], np.uint32)), w).astype(mx.float32))
    W = decode(host["lo12"], host["hi1"], host["i2"], host["gain"], host["c1"], host["c2"])
    ref = np.array(mx.array(W[[2, 0]]).astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_array_equal(out, ref)
