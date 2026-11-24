from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from ink2q.mini_ref import (
    bootstrap_ci,
    fp32_static_convs,
    gate_layers,
    score_finals,
    trace_virtual_layers,
)


def test_gate_layers_checks_all_virtual_layers():
    rng = np.random.default_rng(7)
    ref = rng.standard_normal((48, 8, 64)).astype(np.float32)
    floor = ref + rng.normal(0.0, 1e-2, size=ref.shape).astype(np.float32)
    good = ref + rng.normal(0.0, 1e-2, size=ref.shape).astype(np.float32)
    res = gate_layers(ref, floor, good, max_ratio=1.5)
    assert res["pass"] is True
    assert len(res["layers"]) == 48
    bad = good.copy()
    bad[30] += 0.25
    res_bad = gate_layers(ref, floor, bad, max_ratio=1.5)
    assert res_bad["pass"] is False
    assert res_bad["layers"][30]["ratio"] > 1.5


def test_bootstrap_ci_bounds_zero_diff():
    vals = np.zeros(128, dtype=np.float64)
    lo, hi = bootstrap_ci(vals, seed=0)
    assert lo == 0.0 and hi == 0.0
    pos = np.full(128, 0.001, dtype=np.float64)
    lo2, hi2 = bootstrap_ci(pos, seed=0)
    assert abs(lo2 - 0.001) < 1e-9 and abs(hi2 - 0.001) < 1e-9


def test_score_finals_identical_is_zero_kl(tmp_path: Path):
    rng = np.random.default_rng(11)
    head = mx.array(rng.standard_normal((64, 32)).astype(np.float32)).astype(mx.bfloat16)
    finals = mx.array(rng.standard_normal((16, 32)).astype(np.float32)).astype(mx.bfloat16)
    ids = np.arange(16, dtype=np.int32)
    res = score_finals(head, ids, finals, finals, vocab_real=60)
    assert res["kl"] == 0.0
    assert res["kl_ci95_upper"] == 0.0
    assert res["top1"] == 1.0
    assert res["pass"] is True

    perturbed = (finals.astype(mx.float32) + 5.0).astype(mx.bfloat16)
    res_bad = score_finals(head, ids, finals, perturbed, vocab_real=60)
    assert res_bad["kl"] > 0.0
    assert res_bad["pass"] is False


def test_trace_virtual_layers_captures_looped_visits_and_restores_hooks():
    class BaseLayer:
        def __init__(self, delta: float):
            self.delta = delta

        def __call__(self, h, block_states):
            return (h.astype(mx.float32) + self.delta).astype(mx.bfloat16), block_states

    class SlidingStub(BaseLayer):
        pass

    class FullStub(BaseLayer):
        pass

    class FakeTextModel:
        def __init__(self):
            # 1 prelude, 2 core layers looped 3 times, 1 postlude -> 8 virtual layers
            self.layers = [SlidingStub(1.0), SlidingStub(2.0), FullStub(4.0), FullStub(8.0)]

        def __call__(self, tokens):
            b, t = tokens.shape
            h = mx.zeros((b, t, 4), dtype=mx.bfloat16)
            bs = None
            h, bs = self.layers[0](h, bs)
            pre_h, pre_bs = h, bs
            for _ in range(3):
                h, bs = pre_h, pre_bs
                for layer in self.layers[1:3]:
                    h, bs = layer(h, bs)
            h, bs = self.layers[3](h, bs)
            return (h.astype(mx.float32) * 0.5).astype(mx.bfloat16)

    orig_base_call = BaseLayer.__call__
    lm = FakeTextModel()
    fake_model = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    ids = np.array([1, 2, 3, 4, 5], dtype=np.int32)

    traced, finals = trace_virtual_layers(fake_model, ids)

    assert BaseLayer.__call__ is orig_base_call
    assert traced.shape == (8, 5, 4)
    assert traced.dtype == np.float32
    expected_vals = [1.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0, 15.0]
    for v, exp in enumerate(expected_vals):
        assert np.allclose(traced[v], exp)
    assert finals.shape == (5, 4)
    assert finals.dtype == mx.bfloat16
    assert np.allclose(np.array(finals.astype(mx.float32)), 7.5)


def test_fp32_static_convs_hooks_and_restores():
    from mlx_lm.models.inkling2 import layers as ik_layers

    orig = ik_layers.ShortConvolution.__call__
    conv = ik_layers.ShortConvolution(dims=8, kernel_size=4)
    conv.conv1d.weight = mx.ones((8, 4, 1), dtype=mx.bfloat16) * mx.array(0.1, dtype=mx.bfloat16)
    x = mx.ones((1, 6, 8), dtype=mx.bfloat16)

    with fp32_static_convs():
        assert ik_layers.ShortConvolution.__call__ is not orig
        out, win = conv(x, None, None)
        mx.eval(out, win)
        assert out.shape == x.shape
        assert out.dtype == mx.bfloat16
        assert win.shape == (1, 9, 8)

    assert ik_layers.ShortConvolution.__call__ is orig
