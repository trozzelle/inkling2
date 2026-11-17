"""Per-tensor access to a sharded safetensors checkpoint (the MXFP8 source)."""
import json
from pathlib import Path

import mlx.core as mx


class SourceIndex:
    def __init__(self, root: Path):
        self.root = Path(root)
        index = json.loads((self.root / "model.safetensors.index.json").read_text())
        self.weight_map: dict[str, str] = index["weight_map"]

    def config(self) -> dict:
        return json.loads((self.root / "config.json").read_text())

    def layer_names(self, layer: int) -> list[str]:
        p = f"model.llm.layers.{layer}."
        names = [k for k in self.weight_map if k.startswith(p)]
        names.append(f"model.llm.attn_res.mlp_queries.{layer}")
        if layer > 0:
            names.append(f"model.llm.attn_res.attn_queries.{layer - 1}")
        return sorted(names)

    def global_names(self, final: bool = True) -> list[str]:
        names = ["model.llm.embed.weight", "model.llm.embed_norm.weight"]
        if final:
            names += ["model.llm.attn_res.output_query", "model.llm.norm.weight", "model.llm.unembed.weight"]
        return names

    def shards_for(self, names) -> list[str]:
        return sorted({self.weight_map[n] for n in names})

    def load(self, names) -> dict[str, mx.array]:
        """mx.load of a safetensors file is lazy: only the arrays that are evaluated are read."""
        missing = [n for n in names if n not in self.weight_map]
        if missing:
            raise KeyError(missing)
        by_shard: dict[str, list[str]] = {}
        for n in names:
            by_shard.setdefault(self.weight_map[n], []).append(n)
        out = {}
        for shard, keys in by_shard.items():
            tensors = mx.load(str(self.root / shard))
            out.update({k: tensors[k] for k in keys})
        return out
