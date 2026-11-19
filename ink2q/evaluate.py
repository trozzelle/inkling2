"""Score a variant's final hidden states against the teacher's through the source head (Baekpica _score_heads)."""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q.score import head_logits, score_sequence, slice_of, summarize
from ink2q.source import SourceIndex
from ink2q.tokens import load_manifest, load_tokens


def load_head(src: SourceIndex) -> mx.array:
    return src.load(["model.llm.unembed.weight"])["model.llm.unembed.weight"]


def score_dir(head, seqs, manifest, teacher: Path, variant: Path, vocab_real: int, positions: Path | None = None) -> dict:
    rows, pos = [], {}
    for i, ids in enumerate(seqs):
        ft = mx.load(str(teacher / f"seq-{i:05d}.safetensors"))["final"]
        fv = mx.load(str(variant / f"seq-{i:05d}.safetensors"))["final"]
        meta = manifest[i]
        row = score_sequence(
            ids, meta.get("format", "raw") != "raw",
            lambda c0, c1: head_logits(ft[c0:c1], head, vocab_real),
            lambda c0, c1: head_logits(fv[c0:c1], head, vocab_real),
            positions=positions is not None,
        )
        if positions is not None:
            for k, v in row.pop("pos").items():
                pos[f"{i}_{k}"] = v
        rows.append({"i": i, "case": meta.get("case"), "domain": meta.get("domain"),
                     "format": meta.get("format", "raw"), "source": meta.get("source"), **row})
    res = {"rows": rows, "summary": summarize(rows)}
    res["slices"] = {s: summarize([r for r in rows if slice_of(r) == s]) for s in sorted({slice_of(r) for r in rows})}
    per_dom = [summarize([r for r in rows if r["domain"] == d]) for d in sorted({r["domain"] for r in rows}, key=str)]
    res["macro_domain"] = float(np.mean([s.get("kl_gen", s["kl"]) for s in per_dom]))
    if positions is not None:
        positions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(positions, **pos)
    return res


def main_args(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--variant", type=Path)
    ap.add_argument("--out", type=Path, help="result JSON (single variant)")
    ap.add_argument("--variants-root", type=Path, help="score every subdirectory except the teacher's")
    ap.add_argument("--out-dir", type=Path, help="with --variants-root: one <name>.json per variant")
    ap.add_argument("--vocab-real", type=int, default=200058)
    ap.add_argument("--positions", type=Path, help="write per-position kl/top1/gen arrays to this .npz")
    a = ap.parse_args(argv)
    head = load_head(SourceIndex(a.source))
    seqs, manifest = load_tokens(a.tokens), load_manifest(a.manifest)
    if a.variants_root:
        if not a.out_dir:
            ap.error("--variants-root needs --out-dir")
        a.out_dir.mkdir(parents=True, exist_ok=True)
        summaries = {}
        for d in sorted(p for p in a.variants_root.iterdir() if p.is_dir() and p.resolve() != a.teacher.resolve()):
            if not (d / "seq-00000.safetensors").exists():
                continue
            res = score_dir(head, seqs, manifest, a.teacher, d, a.vocab_real)
            (a.out_dir / f"{d.name}.json").write_text(json.dumps(res, indent=1) + "\n")
            summaries[d.name] = res["summary"]
            print(json.dumps({"variant": d.name, "kl": res["summary"]["kl"]}), flush=True)
        return {"summaries": summaries}
    if not a.out:
        ap.error("--out is required")
    res = score_dir(head, seqs, manifest, a.teacher, a.variant or a.teacher, a.vocab_real, a.positions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))
    return res


if __name__ == "__main__":
    main_args()
