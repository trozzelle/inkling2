import numpy as np

from ink2q import tfexperts


def test_pre_rotation_inverts_the_output_rotation():
    rng = np.random.default_rng(0)
    pre = rng.normal(size=(3, 256))
    svh = rng.uniform(0.5, 2, size=256)
    out = tfexperts.fwht(pre) * svh
    assert np.allclose(tfexperts.pre_rotation(out, svh), pre)


def test_capped_swiglu_pairs_adjacent_columns():
    gu = np.array([[1.0, 2.0, -3.0, 50.0]])
    act = tfexperts.capped_swiglu(gu)
    g, u = gu[0, 0::2], gu[0, 1::2]
    want = 4 * np.tanh(g / 4) / (1 + np.exp(-g)) * 25 * np.tanh(u / 25)
    assert np.allclose(act[0], want)
