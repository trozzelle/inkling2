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


def test_moe_rows_sums_each_rows_eight_picks():
    rng = np.random.default_rng(1)
    ey = tfexperts.to_bf16(rng.normal(size=(3 * 8, 256)))  # per-pick outputs are bf16 values, slot = row * 8 + pick
    got = tfexperts.moe_rows(ey, 3)
    want = ey.reshape(3, 8, 256).sum(1) * tfexperts.MOE_WEIGHT
    assert got.shape == (3, 256)
    assert np.all(np.abs(got - want) <= np.abs(want) * 2 ** -7)  # within a bf16 step: the row's picks, none of another row's


def test_errors_big_leaves_out_outputs_near_zero():
    want = np.ones((4, 128))
    want[0, 0] = 1e-6
    got = want.copy()
    got[0, 0] = 2e-6  # a hundred percent off, at a millionth of the rms
    e = tfexperts.errors(got, want)
    assert e["rel_rms"] > 0 and e["rel_rms_big"] == 0.0 and e["max_rel_big"] == 0.0
