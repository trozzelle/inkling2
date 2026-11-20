"""Prompts for the drafter and route studies: studio bench's built-in prompt (rendered with the source chat template,
as the server renders it), held-out chat sequences cut at their first assistant opener, and agentic sessions cut at
their last assistant opener below a length cap. Written as one JSON id list a sequence for tf-inkling-run."""
import argparse
import json
from pathlib import Path

from ink2q.score import MSG_MODEL
from ink2q.tokens import load_manifest, load_tokens

# studio_inference/bench.py DEFAULT_PROMPT (studio-inference 1.0.0), sent as one user message.
BENCH_PROMPT = (
    "Write a Python function normalize_rows(rows) that accepts a list of dictionaries. "
    "Each dictionary can contain an id, a name, and an email. Ignore rows with a missing "
    "or null id, keep the first row for each id, trim surrounding whitespace from string "
    "names, and lowercase string email addresses. Preserve input order and leave the "
    "original dictionaries unchanged. Return a list of normalized dictionaries. "
    "Include two short tests covering duplicates and missing values. Return code only."
)


def chat_prefix(ids) -> list[int] | None:
    ids = [int(t) for t in ids]
    return ids[: ids.index(MSG_MODEL) + 1] if MSG_MODEL in ids else None


def agentic_prefix(ids, cap: int) -> list[int] | None:
    ids = [int(t) for t in ids]
    cut = [j for j, t in enumerate(ids[:cap]) if t == MSG_MODEL]
    return ids[: cut[-1] + 1] if cut else None


def bench_ids(source: Path) -> list[int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(source))
    text = tok.apply_chat_template([{"role": "user", "content": BENCH_PROMPT}], add_generation_prompt=True,
                                   tokenize=False)
    return [int(t) for t in tok(text, add_special_tokens=False)["input_ids"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", type=Path, required=True)
    ap.add_argument("--heldout-manifest", type=Path, required=True)
    ap.add_argument("--agentic", type=Path, required=True)
    ap.add_argument("--agentic-manifest", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--agentic-count", type=int, default=8)
    ap.add_argument("--cap", type=int, default=4096)
    ap.add_argument("--min-agentic", type=int, default=512)
    a = ap.parse_args(argv)
    prompts = [("bench", "normalize_rows", bench_ids(a.source))]
    for ids, m in zip(load_tokens(a.heldout), load_manifest(a.heldout_manifest)):
        p = chat_prefix(ids)
        if p is not None and len(p) <= a.cap:
            prompts.append(("chat", m.get("case") or m.get("domain"), p))
    taken = 0
    for ids, m in zip(load_tokens(a.agentic), load_manifest(a.agentic_manifest)):
        p = agentic_prefix(ids, a.cap)
        if p is not None and len(p) >= a.min_agentic and taken < a.agentic_count:
            prompts.append(("agentic", m.get("case") or m.get("domain") or "session", p))
            taken += 1
    out = a.out / "ids"
    out.mkdir(parents=True, exist_ok=True)
    paths, meta = [], []
    for i, (kind, case, ids) in enumerate(prompts):
        p = (out / f"seq-{i:05d}.json").resolve()
        p.write_text(json.dumps(ids))
        paths.append(str(p))
        meta.append({"seq": i, "set": kind, "case": case, "length": len(ids)})
    (out / "index.txt").write_text("\n".join(paths) + "\n")
    (a.out / "prompts.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({"prompts": len(meta), "sets": {k: sum(1 for m in meta if m["set"] == k) for k in ("bench", "chat", "agentic")},
                      "max_length": max(m["length"] for m in meta)}))


if __name__ == "__main__":
    main()
