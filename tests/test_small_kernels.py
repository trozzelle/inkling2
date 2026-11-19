import mlx.core as mx
import numpy as np
import pytest

from ink2q.vq import lattice, trellis


@pytest.mark.parametrize("mod,kw", [(lattice, {}), (trellis, {"k": 2}), (trellis, {"k": 3})])
def test_gemv_matches_reference(mod, kw):
    rng = np.random.default_rng(0)
    E, N, K, T, S = 3, 32, 512, 2, 2
    pay = mod.random_payload(rng, E, N, K, **kw)
    x = mx.array(rng.standard_normal((T, K)).astype(np.float32)).astype(mx.bfloat16)
    idx = rng.integers(0, E, size=(T, S)).astype(np.uint32)
    y = np.array(mod.gemv(x, mx.array(idx), pay, N=N, K=K, **kw))
    W = np.stack([mod.decode_ref(pay, e, N, K, **kw) for e in range(E)])
    xr = np.array(x.astype(mx.float32), np.float64)
    ref = np.einsum("tk,tsnk->tsn", xr, W[idx].astype(np.float64))
    scale = np.abs(ref).max()
    np.testing.assert_allclose(y, ref, rtol=2e-3, atol=2e-3 * scale)
