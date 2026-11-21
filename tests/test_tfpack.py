import json
import struct

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from ink2q import tfpack

SHARD_A = {
    "model.llm.embed.weight": np.arange(8, dtype=np.float32).reshape(2, 4),
    "model.llm.norm.weight": np.ones(4, np.float32),
    "model.llm.attn_res.output_query": np.full(4, 0.5, np.float32),
    "model.llm.attn_res.mlp_queries.0": np.full(4, 1.5, np.float32),
    "model.llm.layers.0.attn.wq_du.weight": np.arange(12, dtype=np.int16).reshape(3, 4),
    "model.llm.layers.0.mlp.experts.w13_weight": np.zeros((2, 3), np.uint8),
}
SHARD_B = {
    "model.llm.attn_res.mlp_queries.1": np.full(4, 2.5, np.float32),
    "model.llm.attn_res.attn_queries.0": np.full(4, 3.5, np.float32),
    "model.llm.layers.1.attn.wq_du.weight": np.arange(12, 24, dtype=np.int16).reshape(3, 4),
    "model.visual.backbone.ln_pre.weight": np.ones(4, np.float32),
}


def make_source(root):
    root.mkdir()
    save_file(SHARD_A, str(root / "a.safetensors"))
    save_file(SHARD_B, str(root / "b.safetensors"))
    wm = {k: "a.safetensors" for k in SHARD_A} | {k: "b.safetensors" for k in SHARD_B}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    cfg = {"model_type": "inkling2", "quantization_config": {"quant_method": "fp8"},
           "text_config": {"num_hidden_layers": 2}}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "tokenizer.json").write_text("{}")
    return root


def make_pack(root, status="complete"):
    root.mkdir()
    (root / "checkpoint.json").write_text(json.dumps({"format": tfpack.PACK_FORMAT, "status": status}))
    return root


def test_owner_layer_puts_attention_queries_on_the_next_layer():
    assert tfpack.owner_layer("model.llm.layers.7.attn.wq_du.weight") == 7
    assert tfpack.owner_layer("model.llm.attn_res.mlp_queries.3") == 3
    assert tfpack.owner_layer("model.llm.attn_res.attn_queries.3") == 4
    assert tfpack.owner_layer("model.llm.unembed.weight") is None


def test_dense_file_holds_every_non_expert_backbone_tensor_byte_for_byte(tmp_path):
    src, pack, out = make_source(tmp_path / "src"), make_pack(tmp_path / "pack"), tmp_path / "out"
    res = tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(out)])
    got = load_file(str(out / "model-dense.safetensors"))
    want = {k: v for k, v in (SHARD_A | SHARD_B).items() if ".experts." not in k and not k.startswith("model.visual")}
    assert set(got) == set(want) and res["tensors"] == len(want)
    for k, v in want.items():
        assert got[k].dtype == v.dtype and np.array_equal(got[k], v)
    with open(out / "model-dense.safetensors", "rb") as f:
        assert struct.unpack("<Q", f.read(8))[0] % 8 == 0
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["quantization_config"] == {"quant_method": "exl3", "format": tfpack.PACK_FORMAT}
    assert (out / "experts").resolve() == pack.resolve() and (out / "tokenizer.json").exists()


def test_layer_range_keeps_globals_and_drops_other_layers(tmp_path):
    src, pack, out = make_source(tmp_path / "src"), make_pack(tmp_path / "pack"), tmp_path / "out"
    tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(out), "--layers", "0-0"])
    got = set(load_file(str(out / "model-dense.safetensors")))
    assert "model.llm.attn_res.mlp_queries.0" in got and "model.llm.embed.weight" in got
    assert "model.llm.attn_res.attn_queries.0" not in got          # feeds layer 1
    assert "model.llm.layers.1.attn.wq_du.weight" not in got


def test_incomplete_pack_is_refused(tmp_path):
    src, pack = make_source(tmp_path / "src"), make_pack(tmp_path / "pack", status="partial")
    with pytest.raises(SystemExit):
        tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(tmp_path / "out")])


def test_draft_head_is_copied_when_the_source_has_one(tmp_path):
    src, pack, out = make_source(tmp_path / "src"), make_pack(tmp_path / "pack"), tmp_path / "out"
    save_file({"fuse_weight": np.zeros((6, 8), np.float32)}, str(src / "mtp.safetensors"))
    out.mkdir()
    (out / "mtp.safetensors").symlink_to(src / "mtp.safetensors")                           # an earlier run's link
    tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(out)])
    copy = out / "mtp.safetensors"
    assert not copy.is_symlink() and copy.read_bytes() == (src / "mtp.safetensors").read_bytes()
    save_file({"fuse_weight": np.ones((6, 8), np.float32)}, str(src / "mtp.safetensors"))
    tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(out)])   # a second run replaces it
    assert not copy.is_symlink() and copy.read_bytes() == (src / "mtp.safetensors").read_bytes()


def test_no_draft_head_without_one_in_the_source(tmp_path):
    src, pack, out = make_source(tmp_path / "src"), make_pack(tmp_path / "pack"), tmp_path / "out"
    out.mkdir()
    (out / "mtp.safetensors").symlink_to(tmp_path / "gone.safetensors")                    # left by an earlier run
    tfpack.main_args(["--source", str(src), "--experts", str(pack), "--out", str(out)])
    assert not (out / "mtp.safetensors").is_symlink() and not (out / "mtp.safetensors").exists()
