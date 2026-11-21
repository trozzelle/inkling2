"""A TensorFold Inkling 2 EXL3 model directory: the source's non-expert backbone tensors copied byte for byte into
model-dense.safetensors, its config (quantization naming the EXL3 pack) and tokenizer files, the pack linked as
experts/ and, when the source has one, the draft head copied as mtp.safetensors. The draft head is copied, not
linked: a server started by launchd cannot read a source on an external volume (macOS refuses it), so the model
directory must not reach back into the source."""
import argparse
import json
import re
import shutil
import struct
from pathlib import Path

PACK_FORMAT = "kortexa-inkling2-exl3-v1"
MTP_FILE = "mtp.safetensors"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                   "special_tokens_map.json")
_LAYER = re.compile(r"^model\.llm\.layers\.(\d+)\.")
_MLP_QUERY = re.compile(r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$")
_ATTN_QUERY = re.compile(r"^model\.llm\.attn_res\.attn_queries\.(\d+)$")
_CHUNK = 64 << 20


def owner_layer(name: str) -> int | None:
    """The backbone layer a tensor belongs to, None for a global one (attn_queries.N feeds layer N + 1)."""
    if m := _LAYER.match(name) or _MLP_QUERY.match(name):
        return int(m.group(1))
    if m := _ATTN_QUERY.match(name):
        return int(m.group(1)) + 1
    return None


def wanted(name: str, lo: int, hi: int) -> bool:
    """Text backbone tensors other than routed experts, of layers lo..hi and the globals."""
    if not name.startswith("model.llm.") or ".mlp.experts." in name:
        return False
    layer = owner_layer(name)
    return layer is None or lo <= layer <= hi


def _header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def write_dense(src: Path, names: list[str], out: Path) -> int:
    """`names` from the source shards into one safetensors file, bytes unchanged, in a few MiB of memory; returns
    the data bytes."""
    weight_map = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    headers: dict[str, tuple[dict, int]] = {}
    entries, at = {}, 0
    order = sorted(names)
    for name in order:
        shard = weight_map[name]
        if shard not in headers:
            headers[shard] = _header(src / shard)
        e = headers[shard][0][name]
        size = e["data_offsets"][1] - e["data_offsets"][0]
        entries[name] = {"dtype": e["dtype"], "shape": e["shape"], "data_offsets": [at, at + size]}
        at += size
    head = json.dumps({"__metadata__": {"format": "pt"}, **entries}, separators=(",", ":")).encode()
    head += b" " * (-len(head) % 8)
    with open(out, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for name in order:
            meta, start = headers[weight_map[name]]
            b0, b1 = meta[name]["data_offsets"]
            with open(src / weight_map[name], "rb") as g:
                g.seek(start + b0)
                left = b1 - b0
                while left:
                    buf = g.read(min(_CHUNK, left))
                    f.write(buf)
                    left -= len(buf)
    return at


def main_args(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--experts", type=Path, required=True, help="the EXL3 pack: checkpoint.json and layers/")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", help="first-last backbone layers to include (default: all)")
    a = ap.parse_args(argv)
    cfg = json.loads((a.source / "config.json").read_text())
    pack = json.loads((a.experts / "checkpoint.json").read_text())
    if pack.get("format") != PACK_FORMAT or pack.get("status") != "complete":
        raise SystemExit(f"{a.experts}: not a complete {PACK_FORMAT} pack")
    n = cfg["text_config"]["num_hidden_layers"]
    lo, hi = map(int, a.layers.split("-")) if a.layers else (0, n - 1)
    weight_map = json.loads((a.source / "model.safetensors.index.json").read_text())["weight_map"]
    names = [k for k in weight_map if wanted(k, lo, hi)]
    a.out.mkdir(parents=True, exist_ok=True)
    data = write_dense(a.source, names, a.out / "model-dense.safetensors")
    cfg["quantization_config"] = {"quant_method": "exl3", "format": PACK_FORMAT}
    (a.out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    copied = [f for f in TOKENIZER_FILES if (a.source / f).exists()]
    for f in copied:
        shutil.copyfile(a.source / f, a.out / f)
    link = a.out / "experts"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(a.experts.resolve(), target_is_directory=True)
    mtp = a.out / MTP_FILE
    if mtp.is_symlink() or mtp.exists():
        mtp.unlink()
    if (a.source / MTP_FILE).exists():
        tmp = a.out / f".{MTP_FILE}.tmp"
        shutil.copyfile(a.source / MTP_FILE, tmp)
        tmp.replace(mtp)
    res = {"tensors": len(names), "dense_bytes": data, "layers": [lo, hi], "tokenizer": copied}
    (a.out / "inkling2-pack.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return res


if __name__ == "__main__":
    main_args()
