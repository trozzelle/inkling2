"""Download only the source shards holding the given layers (plus the globals the reference reads) into a partial snapshot."""
import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

from ink2q.config import SOURCE_REPO, SOURCE_REV
from ink2q.source import SourceIndex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", required=True, help="first-last, e.g. 0-11")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for f in ("model.safetensors.index.json", "config.json"):
        hf_hub_download(SOURCE_REPO, f, revision=SOURCE_REV, local_dir=a.out)
    src = SourceIndex(a.out)
    lo, hi = map(int, a.layers.split("-"))
    globals_ = [n for n in src.global_names(final=True) if n != "model.llm.unembed.weight"]  # R.Globals reads norm
    names = globals_ + [n for l in range(lo, hi + 1) for n in src.layer_names(l)]
    shards = src.shards_for(names)
    print(json.dumps({"shards": shards}), flush=True)
    for s in shards:
        hf_hub_download(SOURCE_REPO, s, revision=SOURCE_REV, local_dir=a.out)
        print(json.dumps({"downloaded": s}), flush=True)


if __name__ == "__main__":
    main()
