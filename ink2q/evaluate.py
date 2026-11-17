"""Score a variant's final hidden states against the teacher's through the source head (Baekpica _score_heads)."""
import argparse
import json
from pathlib import Path

import mlx.core as mx

from ink2q.score import head_logits, score_sequence, summarize
from ink2q.source import SourceIndex
from ink2q.tokens import load_manifest, load_tokens


def load_head(src: SourceIndex) -> mx.array:
    return src.load(["model.llm.unembed.weight"])["model.llm.unembed.weight"]


def main_args(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--variant", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab-real", type=int, default=200058)
    a = ap.parse_args(argv)
    variant = a.variant or a.teacher
    head = load_head(SourceIndex(a.source))
    seqs, manifest = load_tokens(a.tokens), load_manifest(a.manifest)
    rows = []
    for i, ids in enumerate(seqs):
        ft = mx.load(str(a.teacher / f"seq-{i:05d}.safetensors"))["final"]
        fv = mx.load(str(variant / f"seq-{i:05d}.safetensors"))["final"]
        meta = manifest[i]
        row = score_sequence(
            ids, meta.get("format", "raw") != "raw",
            lambda c0, c1: head_logits(ft[c0:c1], head, a.vocab_real),
            lambda c0, c1: head_logits(fv[c0:c1], head, a.vocab_real),
        )
        rows.append({"i": i, "case": meta.get("case"), "domain": meta.get("domain"), **row})
    res = {"rows": rows, "summary": summarize(rows)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    return res


if __name__ == "__main__":
    main_args()
