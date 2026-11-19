"""Indexes over converted expert files (oMLX layout: model.layers.{l}.mlp.experts.{w13,w2}.{weight,scales,biases})."""
import json
import os
from pathlib import Path


def expert_file(convert_dir: Path, layer: int, which: str, bits: int) -> Path:
    return Path(convert_dir) / "experts" / f"L{layer:02d}" / f"{which}_b{bits}.safetensors"


def write_expert_index(alloc: dict, convert_dir: Path, out: Path, layers=None, group_size: int = 128) -> None:
    """An oMLX-layout directory (config.json quantization dict and index, files symlinked) holding the converted
    experts of `alloc` = {"<layer>": {"w13": bits, "w2": bits}}, restricted to `layers` if given."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    q = {"bits": 3, "group_size": group_size, "mode": "affine"}
    wmap = {}
    for l_str, widths in sorted(alloc.items(), key=lambda kv: int(kv[0])):
        layer = int(l_str)
        if layers is not None and layer not in layers:
            continue
        for w in ("w13", "w2"):
            bits = widths[w]
            key = f"model.layers.{layer}.mlp.experts.{w}"
            q[key] = {"bits": bits, "group_size": group_size, "mode": "affine"}
            src = expert_file(convert_dir, layer, w, bits).resolve()
            rel = Path(f"L{layer:02d}") / src.name
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            if (out / rel).is_symlink() or (out / rel).exists():
                (out / rel).unlink()
            os.symlink(src, out / rel)
            for k in ("weight", "scales", "biases"):
                wmap[f"{key}.{k}"] = str(rel)
    (out / "config.json").write_text(json.dumps({"quantization": q}, indent=1) + "\n")
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wmap}, indent=1) + "\n")
