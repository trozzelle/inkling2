"""Baseline B: teacher-forced score of an oMLX-loaded Inkling 2 build against the MXFP8 teacher's final states.

Runs in the oMLX venv with PYTHONPATH set to this repo (ink2q.score, ink2q.source, ink2q.tokens and
ink2q.memguard import no mlx-lm). At the pinned oMLX commit 624cff0e, `model.model(ids)` returns the final
normalized hidden states and `model.logits_from_hidden` gives float32 logits over the padded vocabulary,
softcapped at 30 with -inf on the padding (omlx/patches/inkling2/model.py); logits are built per 1024-position
chunk and sliced to the 200,058 real tokens.
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from omlx.patches.inkling2 import vlm

from ink2q.memguard import Floor, apply_limits
from ink2q.score import head_logits, score_sequence, summarize
from ink2q.source import SourceIndex
from ink2q.tokens import load_manifest, load_tokens

VOCAB_REAL = 200058


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--teacher", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    apply_limits(236, 4)  # the 213 GiB build must be resident; the floor below still guards the host
    floor = Floor(8)
    t0 = time.monotonic()
    model, _ = vlm.load(str(a.model), lazy=True)
    done = 0
    for k, p in tree_flatten(model.parameters()):
        mx.eval(p)
        done += p.nbytes
        if done // (8 << 30) != (done - p.nbytes) // (8 << 30):  # every 8 GiB loaded
            print(json.dumps({"loaded_gib": round(done / 2**30, 1), "active_gib": round(mx.get_active_memory() / 2**30, 1),
                              "peak_gib": round(mx.get_peak_memory() / 2**30, 1), "last": k}), flush=True)
            floor.check(f"load {done / 2**30:.0f} GiB")
    load_s = time.monotonic() - t0
    floor.check("model loaded")
    head = SourceIndex(a.source).load(["model.llm.unembed.weight"])["model.llm.unembed.weight"]
    seqs, manifest = load_tokens(a.tokens), load_manifest(a.manifest)
    rows = []
    for i, ids in enumerate(seqs):
        ft = mx.load(str(a.teacher / f"seq-{i:05d}.safetensors"))["final"]
        # final hidden states only; logits per 1024-position chunk (full [8192, 201024] float32 logits are 6 GiB)
        h = model.model(mx.array(ids)[None])[0]
        mx.eval(h)
        floor.check(f"seq {i}")
        row = score_sequence(ids, manifest[i].get("format", "raw") != "raw",
                             lambda c0, c1: head_logits(ft[c0:c1], head),
                             lambda c0, c1: model.logits_from_hidden(h[c0:c1])[:, :VOCAB_REAL])
        rows.append({"i": i, "case": manifest[i].get("case"), "domain": manifest[i].get("domain"), **row})
        print(json.dumps({"seq": i, "kl": row["kl"] / row["positions"]}), flush=True)
    res = {"rows": rows, "summary": summarize(rows), "load_seconds": round(load_s, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["summary"], indent=1))


if __name__ == "__main__":
    main()
