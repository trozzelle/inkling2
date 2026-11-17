# Copyright © 2026 Apple Inc.
# CONFIG, create_model, _CHECKPOINT_NAMES and checkpoint are copied from mlx-lm
# tests/models/test_inkling2.py at jundot/mlx-lm@25c8b4322c5dad4bca0887a6d878d0ffa2655928 (MIT).
import json
import re
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm.models import inkling2
from mlx_lm.utils import quantize_model

CONFIG = {
    "model_type": "inkling2",
    "image_token_id": 127,
    "text_config": {
        "hidden_size": 64,
        "num_hidden_layers": 8,
        "vocab_size": 128,
        "unpadded_vocab_size": 120,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "swa_num_attention_heads": 4,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "sliding_window_size": 8,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "d_rel": 4,
        "rel_extent": 8,
        "log_scaling_n_floor": 4,
        "log_scaling_alpha": 0.1,
        "rms_norm_eps": 1e-6,
        "qk_l2_clip_threshold": 3.0,
        "rel_l2_clip_threshold": 2.0,
        "sconv_kernel_size": 4,
        "n_layers_per_attn_res_block": 3,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 64,
        "mlp_gate_softcap": 4.0,
        "mlp_up_softcap": 25.0,
        "route_scale": 2.0,
        "final_logit_softcapping": 30.0,
    },
}


def create_model() -> inkling2.Model:
    mx.random.seed(0)
    model = inkling2.Model(inkling2.ModelArgs.from_dict(CONFIG))
    weights = []
    for k, v in tree_flatten(model.parameters()):
        if k.endswith(("norm.weight", "l2_clip.scale", "lambda_1", "lambda_2")):
            v = 1 + 0.1 * mx.random.normal(v.shape)
        else:
            fan_in = v.shape[-1] if v.ndim > 1 else 16
            v = mx.random.normal(v.shape) * fan_in**-0.5
        weights.append((k, v))
    model.update(tree_unflatten(weights))
    mx.eval(model.parameters())
    return model


_CHECKPOINT_NAMES = [
    (r"^lm_head\.", "model.llm.unembed."),
    (r"^model\.language_model\.embed_tokens\.", "model.llm.embed."),
    (r"^model\.language_model\.", "model.llm."),
    (r"\.final_input\.query$", ".attn_res.output_query"),
    (r"\.layers\.(\d+)\.mlp_input\.query$", r".attn_res.mlp_queries.\1"),
    (r"\.self_attn\.q_proj\.", ".attn.wq_du."),
    (r"\.self_attn\.k_proj\.", ".attn.wk_dv."),
    (r"\.self_attn\.v_proj\.", ".attn.wv_dv."),
    (r"\.self_attn\.r_proj\.", ".attn.wr_du."),
    (r"\.self_attn\.o_proj\.", ".attn.wo_ud."),
    (r"\.self_attn\.gate_proj\.", ".attn.gate_attn."),
    (r"\.self_attn\.sink_proj\.", ".attn.attn_sink_proj."),
    (r"\.self_attn\.", ".attn."),
    (r"\.post_attention_convolution\.", ".attn_sconv."),
    (r"\.post_mlp_convolution\.", ".mlp_sconv."),
    (r"\.conv1d\.weight$", ".weight"),
    (r"sconv\.static_weight$", "sconv.static_w"),
    (r"sconv\.weight_proj\.weight$", "sconv.weight_proj"),
    (r"\.post_attention_layernorm\.", ".mlp_norm."),
    (r"\.input_layernorm\.", ".attn_norm."),
    (r"\.(q|k|v)_input_layernorm\.", r".\1_input_norm."),
    (r"\.mlp\.gate\.e_score_correction_bias$", ".mlp.gate.bias"),
    (r"^model\.vision_tower\.pre_layernorm\.", "model.visual.backbone.ln_pre."),
    (r"^model\.vision_tower\.merger\.linear_1\.", "model.visual.proj_in."),
    (r"^model\.vision_tower\.merger\.linear_2\.", "model.visual.proj_out."),
    (r"^model\.vision_tower\.", "model.visual.backbone."),
    (r"position_embedding\.weight$", "position_embedding"),
    (r"\.layernorm_before\.", ".ln_1."),
    (r"\.layernorm_after\.", ".ln_2."),
    (r"\.mlp\.fc1\.", ".mlp.c_fc."),
    (r"\.mlp\.fc2\.", ".mlp.c_proj."),
    (r"\.lambda_(\d)$", r".ls_\1.gamma"),
]


def checkpoint(model: inkling2.Model) -> dict:
    weights = dict(tree_flatten(model.parameters()))
    ckpt = {}
    denominator = mx.array([7.5])
    for k, v in weights.items():
        if m := re.match(r"(.+\.mlp\.experts)\.(gate_up|down)_proj\.(\w+)$", k):
            prefix, name, param = m.groups()
            prefix = prefix.replace("model.language_model.", "model.llm.")
            w = f"{prefix}.{'w13' if name == 'gate_up' else 'w2'}_weight"
            if param == "weight":
                ckpt[w] = v.view(mx.uint8)
            elif param == "scales":
                ckpt[f"{w}_scale"] = v
            else:
                ckpt[f"{w}_amax"] = v * denominator / 2688.0
                ckpt[f"{w}_outer_scale_denominator"] = denominator
            continue
        if m := re.match(r".+\.layers\.(\d+)\.attention_input\.query$", k):
            ckpt[f"model.llm.attn_res.attn_queries.{int(m.group(1)) - 1}"] = v
            continue
        for pattern, replacement in _CHECKPOINT_NAMES:
            k = re.sub(pattern, replacement, k)
        if k.endswith("sconv.weight"):
            v = v.swapaxes(1, 2)
        ckpt[k] = v
    return ckpt


# ---- ink2q additions ----

def quantize_experts_mxfp8(model) -> None:
    def predicate(path, module):
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"} if ".mlp.experts." in path else False

    quantize_model(model, {}, None, None, quant_predicate=predicate)


def _owner(name: str) -> str:
    if m := re.match(r"model\.llm\.layers\.(\d+)\.", name):
        return f"model-{int(m.group(1)) + 1:05d}.safetensors"
    if m := re.match(r"model\.llm\.attn_res\.mlp_queries\.(\d+)$", name):
        return f"model-{int(m.group(1)) + 1:05d}.safetensors"
    if m := re.match(r"model\.llm\.attn_res\.attn_queries\.(\d+)$", name):
        return f"model-{int(m.group(1)) + 2:05d}.safetensors"
    return "model-00000.safetensors"


def write_checkpoint(ckpt: dict, root: Path, config: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    shards: dict[str, dict] = {}
    for k, v in ckpt.items():
        shards.setdefault(_owner(k), {})[k] = v
    for fname, tensors in shards.items():
        mx.save_safetensors(str(root / fname), tensors)
    index = {"metadata": {}, "weight_map": {k: _owner(k) for k in ckpt}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    (root / "config.json").write_text(json.dumps(config))
