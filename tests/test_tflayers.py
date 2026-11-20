import mlx.core as mx
import numpy as np

from ink2q import tflayers
from tests.helpers import toy


def test_gate_scores_each_layer_against_its_floor():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3, 4, 8)).astype(np.float32)
    floor = ref + 0.01 * rng.normal(size=ref.shape).astype(np.float32)
    close = ref + 0.012 * rng.normal(size=ref.shape).astype(np.float32)
    far = close.copy()
    far[2] = ref[2] + 0.05 * rng.normal(size=ref[2].shape).astype(np.float32)
    assert tflayers.gate(ref, floor, close)["pass"]
    res = tflayers.gate(ref, floor, far)
    assert not res["pass"] and res["layers"][2]["ratio"] > 1.5 >= res["layers"][1]["ratio"]


def test_read_dump_reshapes_bf16_layers(tmp_path):
    x = mx.random.normal((2, 3, 4)).astype(mx.bfloat16)
    np.array(x.view(mx.uint16)).astype("<u2").tofile(tmp_path / "d.bin")
    got = tflayers.read_dump(tmp_path / "d.bin", tokens=3, hidden=4)
    assert got.shape == (2, 3, 4) and np.array_equal(got, np.array(x.astype(mx.float32)))


def test_fp32_convs_changes_a_layer_slightly_and_restores_the_reference():
    m = toy.create_model()
    m.set_dtype(mx.bfloat16)
    x = (mx.random.normal((1, 12, 64)) * 0.5).astype(mx.bfloat16)
    layer = m.model.language_model.layers[0]
    base, _ = layer(x, None)
    with tflayers.fp32_convs():
        alt, _ = layer(x, None)
    again, _ = layer(x, None)
    assert mx.array_equal(base, again)                     # the patch does not leak
    d = tflayers.rel_rms(np.array(alt.astype(mx.float32)), np.array(base.astype(mx.float32)))
    assert 0 < d < 0.05
