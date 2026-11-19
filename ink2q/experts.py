"""Variant routed experts for one decoder layer of the mlx-lm Inkling 2 model."""
import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, _quantize_experts


def affine_switch(weight, scales, biases, bits, group_size=128, mode="affine") -> QuantizedSwitchLinear:
    """Build a QuantizedSwitchLinear around existing tensors; the constructor would quantize a random weight."""
    m = QuantizedSwitchLinear.__new__(QuantizedSwitchLinear)
    nn.Module.__init__(m)
    m.group_size, m.bits, m.mode = group_size, bits, mode
    m.weight, m.scales, m.biases, m.global_scale = weight, scales, biases, None
    m.freeze()
    return m


@dataclass
class ExpertSpec:
    gate_up: nn.Module
    down: nn.Module
    label: str


def _proj(layer_module, which):
    ex = layer_module.mlp.experts
    return ex.gate_up_proj if which == "gate_up" else ex.down_proj


def source_bf16(layer_module, which: str, experts: slice) -> mx.array:
    p = _proj(layer_module, which)
    b = p.get("biases")
    biases = b[experts] if b is not None else None
    return mx.dequantize(p.weight[experts], p.scales[experts], biases=biases, group_size=p.group_size,
                         bits=p.bits, mode=p.mode).astype(mx.bfloat16)


def _chunked(layer_module, which, fn, chunk):
    n = _proj(layer_module, which).weight.shape[0]
    parts = []
    for e0 in range(0, n, chunk):
        out = fn(source_bf16(layer_module, which, slice(e0, e0 + chunk)))
        mx.eval(out)
        parts.append(out)
    return [mx.concatenate([p[i] for p in parts]) for i in range(len(parts[0]))]


def rtn(layer_module, bits_gu, bits_dn, group_size=128, chunk=32) -> ExpertSpec:
    mods = []
    for which, bits in (("gate_up", bits_gu), ("down", bits_dn)):
        w, s, b = _chunked(layer_module, which, lambda t: mx.quantize(t, group_size=group_size, bits=bits), chunk)
        mods.append(affine_switch(w, s, b, bits, group_size))
    return ExpertSpec(mods[0], mods[1], f"rtn{bits_gu}/{bits_dn}")


def nvfp4(layer_module, chunk=32) -> ExpertSpec:
    """NVFP4 with a per-expert fp32 tensor scale, as the real format and mlx-lm's expert quantizer use."""
    mods = []
    for which in ("gate_up", "down"):
        w, s, gs = _chunked(layer_module, which,
                            lambda t: _quantize_experts(t, group_size=16, bits=4, mode="nvfp4"), chunk)
        m = affine_switch(w, s, None, 4, 16, mode="nvfp4")
        m.global_scale = gs
        mods.append(m)
    return ExpertSpec(mods[0], mods[1], "nvfp4")


def from_mlx_dir(root: Path, layer: int) -> ExpertSpec:
    """Experts of one layer from an oMLX-layout checkpoint (model.layers.{l}.mlp.experts.{w13,w2}.*)."""
    root = Path(root)
    q = json.loads((root / "config.json").read_text())["quantization"]
    wmap = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    mods = []
    for w in ("w13", "w2"):
        key = f"model.layers.{layer}.mlp.experts.{w}"
        spec = q.get(key, {"bits": q["bits"], "group_size": q["group_size"], "mode": q.get("mode", "affine")})
        shard = mx.load(str(root / wmap[f"{key}.weight"]))
        t = {k: shard[f"{key}.{k}"] for k in ("weight", "scales", "biases")}
        mx.eval(list(t.values()))
        del shard
        mods.append(affine_switch(t["weight"], t["scales"], t["biases"], spec["bits"], spec["group_size"],
                                  spec.get("mode", "affine")))
    return ExpertSpec(mods[0], mods[1], f"mlx:{root.name}")


@contextlib.contextmanager
def install(layer_module, spec: ExpertSpec):
    ex = layer_module.mlp.experts
    old = (ex.gate_up_proj, ex.down_proj)
    ex.gate_up_proj, ex.down_proj = spec.gate_up, spec.down
    try:
        yield
    finally:
        ex.gate_up_proj, ex.down_proj = old
