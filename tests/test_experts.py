import json

import mlx.core as mx

from ink2q import experts as X
from tests.helpers.toy import CONFIG, create_model, quantize_experts_mxfp8


def layer_out(layer, x):
    return layer.mlp(x)


def test_rtn_matches_mx_quantize_and_install_restores():
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[2]
    x = mx.random.normal((1, 5, CONFIG["text_config"]["hidden_size"])).astype(mx.bfloat16)
    base = layer_out(L, x)
    spec = X.rtn(L, 4, 4, group_size=32)
    gu = X.source_bf16(L, "gate_up", slice(None))
    wq, s, b = mx.quantize(gu, group_size=32, bits=4)
    assert mx.array_equal(spec.gate_up.weight, wq)
    with X.install(L, spec):
        changed = layer_out(L, x)
    assert not mx.array_equal(changed, base)
    assert mx.array_equal(layer_out(L, x), base)


def test_nvfp4_variant_runs():
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[2]
    x = mx.random.normal((1, 5, CONFIG["text_config"]["hidden_size"])).astype(mx.bfloat16)
    base = layer_out(L, x)
    with X.install(L, X.nvfp4(L)):
        y = layer_out(L, x)
    rel = float(mx.linalg.norm((y - base).astype(mx.float32)) / mx.linalg.norm(base.astype(mx.float32)))
    assert 0 < rel < 0.5


def test_from_mlx_dir_reads_bits_and_tensors(tmp_path):
    m = create_model()
    quantize_experts_mxfp8(m)
    L = m.model.language_model.layers[1]
    spec = X.rtn(L, 2, 3, group_size=32)
    tensors = {}
    for w, mod in (("w13", spec.gate_up), ("w2", spec.down)):
        for k in ("weight", "scales", "biases"):
            tensors[f"model.layers.1.mlp.experts.{w}.{k}"] = getattr(mod, k)
    mx.save_safetensors(str(tmp_path / "model-00001.safetensors"), tensors)
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model-00001.safetensors" for k in tensors}}))
    q = {"bits": 2, "group_size": 32, "mode": "affine",
         "model.layers.1.mlp.experts.w13": {"bits": 2, "group_size": 32, "mode": "affine"},
         "model.layers.1.mlp.experts.w2": {"bits": 3, "group_size": 32, "mode": "affine"}}
    (tmp_path / "config.json").write_text(json.dumps({"quantization": q}))
    got = X.from_mlx_dir(tmp_path, 1)
    assert got.gate_up.bits == 2 and got.down.bits == 3 and got.gate_up.group_size == 32
    assert mx.array_equal(got.down.weight, spec.down.weight)
