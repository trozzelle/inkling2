import mlx.core as mx
import pytest

from ink2q.source import SourceIndex
from tests.helpers.toy import CONFIG, checkpoint, create_model, quantize_experts_mxfp8, write_checkpoint


@pytest.fixture(scope="module")
def toy_source(tmp_path_factory):
    model = create_model()
    quantize_experts_mxfp8(model)
    ckpt = checkpoint(model)
    root = tmp_path_factory.mktemp("src")
    write_checkpoint(ckpt, root, CONFIG)
    return root, ckpt


def test_layer_names(toy_source):
    root, ckpt = toy_source
    src = SourceIndex(root)
    names = src.layer_names(3)
    assert "model.llm.attn_res.mlp_queries.3" in names
    assert "model.llm.attn_res.attn_queries.2" in names
    assert "model.llm.layers.3.mlp.experts.w13_weight" in names
    assert all(n.startswith(("model.llm.layers.3.", "model.llm.attn_res.")) for n in names)
    assert "model.llm.attn_res.attn_queries.0" not in src.layer_names(0)


def test_load_selected(toy_source):
    root, ckpt = toy_source
    src = SourceIndex(root)
    names = src.layer_names(2)
    got = src.load(names)
    assert set(got) == set(names)
    for k in names:
        assert mx.array_equal(got[k], ckpt[k])
    with pytest.raises(KeyError):
        src.load(["model.llm.nope"])
