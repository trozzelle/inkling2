import mlx.core as mx
import numpy as np

from ink2q import collect as C
from tests.helpers.toy import CONFIG, create_model, quantize_experts_mxfp8


def test_gram_bf16x3_matches_fp64():
    rng = np.random.default_rng(0)
    u = rng.standard_normal((3000, 96)).astype(np.float32)
    got = np.array(C.gram_bf16x3(mx.array(u)), np.float64)
    want = u.astype(np.float64).T @ u.astype(np.float64)
    assert np.abs(got - want).max() / np.abs(want).max() < 2e-5


def test_expert_stats_match_numpy():
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[2]
    H = CONFIG["text_config"]["hidden_size"]
    x = mx.random.normal((40, H), key=mx.random.key(1)).astype(mx.bfloat16)
    st = C.ExpertStats(L, experts=None, raw=True, normalized=True)
    st.add(x)
    host = st.to_host()
    inds, _ = L.mlp.gate(x)
    inds = np.array(inds)
    xn = np.array(x.astype(mx.float32), np.float64)
    for e in range(CONFIG["text_config"]["n_routed_experts"]):
        rows = np.where((inds == e).any(-1))[0]
        assert host["count"][e] == len(rows)
        xe = xn[rows]
        np.testing.assert_allclose(host["xx_raw"][e], xe.T @ xe, rtol=1e-5, atol=1e-5)
        u = xe / np.sqrt((xe**2).mean(-1, keepdims=True))
        want = u.T @ u          # the split omits lo·lo: error ≈ 2^-16 of the matrix scale
        assert np.abs(host["xx_norm"][e] - want).max() / np.abs(want).max() < 2e-5


def test_to_host_covers_only_tracked_experts():
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[2]
    x = mx.random.normal((40, CONFIG["text_config"]["hidden_size"]), key=mx.random.key(1)).astype(mx.bfloat16)
    full = C.ExpertStats(L, None, raw=True, normalized=False)
    part = C.ExpertStats(L, [3, 1], raw=True, normalized=False)
    full.add(x); part.add(x)
    hf, hp = full.to_host(np.float32), part.to_host(np.float32)
    assert hp["xx_raw"].shape[0] == 2 and hp["xx_raw"].dtype == np.float32
    assert list(hp["experts"]) == [3, 1]
    np.testing.assert_array_equal(hp["xx_raw"][0], hf["xx_raw"][3])
    np.testing.assert_array_equal(hp["count"], hf["count"][[3, 1]])


def test_activations_match_the_layer_experts():
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[2]
    H = CONFIG["text_config"]["hidden_size"]
    x = mx.random.normal((12, H), key=mx.random.key(2)).astype(mx.bfloat16)
    st = C.ExpertStats(L, experts=[0], raw=True, normalized=False)
    ex = L.mlp.experts
    idx = mx.zeros((12, 1), mx.uint32)
    gu = ex.gate_up_proj(mx.expand_dims(x, (-2, -3)), idx)
    want = ex.act_fn(gu[..., 0::2], gu[..., 1::2]).reshape(12, -1)
    got = st._act(x, 0)
    np.testing.assert_allclose(np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)), rtol=2e-2, atol=2e-2)


def test_capture_returns_mlp_input_and_keeps_output():
    m = create_model()
    L = m.model.language_model.layers[1]
    x = mx.random.normal((1, 6, CONFIG["text_config"]["hidden_size"])).astype(mx.bfloat16)
    want = L.mlp(x)
    with C.capture(L) as sink:
        got = L.mlp(x)
    assert mx.array_equal(sink[0], x) and mx.array_equal(got, want)
    assert not isinstance(L.mlp, C._Capture)
