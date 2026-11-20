"""tf-inkling-run taps output: per sequence the 8 drafter taps (layers 5..41 and the finals), every layer's expert
picks and two layers' MoE inputs a row, beside the tokens; and the check that tap 7 (the finals) predicts each greedy
token, which pins the row alignment."""
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

from ink2q.score import head_logits
from ink2q.source import SourceIndex

H, LAYERS, TOPK, EXPERTS = 4608, 48, 8, 384


def load_seq(d: Path, seq: int) -> dict:
    meta = json.loads((d / f"seq-{seq:05d}.tokens.json").read_text())
    tokens = np.array(meta["tokens"], np.int64)
    T = len(tokens) - 1
    return {
        "prompt": meta["prompt"], "tokens": tokens,
        "taps": np.fromfile(d / f"seq-{seq:05d}.taps.bin", "<u2").reshape(T, 8, H),
        "eids": np.fromfile(d / f"seq-{seq:05d}.eids.bin", "<u4").reshape(T, LAYERS, TOPK),
        "xm": np.fromfile(d / f"seq-{seq:05d}.xm.bin", "<u2").reshape(T, 2, H),
    }


def check_alignment(d: Path, seq: int, source: Path) -> dict:
    s = load_seq(d, seq)
    head = SourceIndex(source).load(["model.llm.unembed.weight"])["model.llm.unembed.weight"]
    rows = np.arange(s["prompt"] - 1, len(s["tokens"]) - 1)
    fin = mx.array(s["taps"][rows, 7]).view(mx.bfloat16)
    pred = np.array(mx.argmax(head_logits(fin, head), axis=-1))
    match = int((pred == s["tokens"][rows + 1]).sum())
    e = s["eids"]
    eids_ok = bool((e < EXPERTS).all() and all(len(set(r)) == TOPK for r in e.reshape(-1, TOPK)[:: max(1, e.shape[0] // 64)]))
    return {"rows": int(len(rows)), "match": match, "rate": match / len(rows), "eids_ok": eids_ok}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--dir", type=Path, required=True)
    c.add_argument("--seq", type=int, default=0)
    c.add_argument("--source", type=Path, required=True)
    a = ap.parse_args(argv)
    res = check_alignment(a.dir, a.seq, a.source)
    print(json.dumps(res))
    sys.exit(0 if res["rate"] >= 0.99 and res["eids_ok"] else 1)


if __name__ == "__main__":
    main()
