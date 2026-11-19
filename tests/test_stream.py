import mlx.core as mx
import numpy as np

from ink2q import model as M
from ink2q import stream as S
from ink2q.source import SourceIndex
from tests.helpers.toy import CONFIG, checkpoint, create_model, quantize_experts_mxfp8, write_checkpoint


def test_streamed_load_matches_full_forward(tmp_path):
    full = create_model()
    quantize_experts_mxfp8(full)
    write_checkpoint(checkpoint(full), tmp_path, CONFIG)
    ids = mx.array(np.random.default_rng(0).integers(0, 120, size=(24,)))
    expected = full(ids[None])[0]

    src = SourceIndex(tmp_path)
    m = M.build(src.config())
    M.load_globals(m, src)
    st = S.embed(m, ids)
    for layer in range(CONFIG["text_config"]["num_hidden_layers"]):
        M.load_layer(m, src, layer)
        st = S.step(m, layer, st)
        mx.eval(st.h, st.blocks)
        M.drop_layer(m, layer)
    got = S.logits(m, S.final_hidden(m, st))
    assert got.shape == expected.shape
    np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-4, rtol=1e-4)


def test_lean_state_matches_full_blocks():
    full = create_model()
    ids = mx.array(np.random.default_rng(1).integers(0, 120, size=(20,)))
    st = S.embed(full, ids)
    ln = S.embed_lean(full, ids)
    for layer in range(CONFIG["text_config"]["num_hidden_layers"]):
        st = S.step(full, layer, st)
        ln = S.step_lean(full, layer, ln)
        assert mx.array_equal(st.h, ln.h)
    assert mx.array_equal(S.final_hidden(full, st), S.final_hidden_lean(full, ln))
