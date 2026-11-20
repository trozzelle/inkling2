"""Offline DSpark acceptance on TensorFold taps. oMLX's own Inkling2DSparkModel (omlx 624cff0e) proposes a 7-token
block at every generated row of each greedy run from the recorded taps, with the target's bf16 embed, embed norm and
unembed. Writes proposals, the first rows' step logits (the Phase 2 reference) and acceptance summaries."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn
from omlx.patches.inkling2.dspark import (Inkling2DSparkConfig, Inkling2DSparkModel, load_inkling2_dspark_model,
                                          wrap_inkling2_dspark)

from ink2q.source import SourceIndex
from ink2q.tfaccept import accept_lengths, per_position, simulate
from ink2q.tftaps import H, load_seq


def stub_target(src: SourceIndex):
    w = src.load(["model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.unembed.weight"])
    v, h = w["model.llm.embed.weight"].shape
    embed = nn.Embedding(v, h)
    embed.weight = w["model.llm.embed.weight"]
    norm = nn.RMSNorm(h, eps=1e-6)
    norm.weight = w["model.llm.embed_norm.weight"]
    head = nn.Linear(h, v, bias=False)
    head.weight = w["model.llm.unembed.weight"]
    return SimpleNamespace(model=SimpleNamespace(embed=embed, embed_norm=norm), lm_head=head)


def draft_model(model_dir: Path, quantized: bool):
    if quantized:
        return load_inkling2_dspark_model(model_dir)
    cfg = Inkling2DSparkConfig.from_target_config(json.loads((model_dir / "config.json").read_text()))
    m = Inkling2DSparkModel(cfg)
    m.load_weights(list(mx.load(str(model_dir / "mtp.safetensors")).items()), strict=True)
    mx.eval(m.parameters())
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True, help="config.json and mtp.safetensors (the source)")
    ap.add_argument("--taps", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quantized", action="store_true", help="oMLX's 8-bit draft layers, as it serves them")
    ap.add_argument("--save-logits", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = draft_model(a.model_dir, a.quantized)
    m.bind(stub_target(SourceIndex(a.model_dir)))
    drafter = wrap_inkling2_dspark(m, str(a.model_dir), fixed=True)
    captured = []
    step = m.markov_head.apply_step_logits

    def capture(base, prev):
        out = step(base, prev)
        captured.append(out)
        return out

    m.markov_head.apply_step_logits = capture
    prompts = json.loads(a.prompts.read_text())
    lengths = {}
    for meta in prompts:
        seq = meta["seq"]
        s = load_seq(a.taps, seq)
        L, tokens, taps = s["prompt"], s["tokens"], s["taps"]
        T = len(tokens) - 1
        props = np.full((T, 7), -1, np.int32)
        saved = []
        st = SimpleNamespace(uid=seq)
        row = drafter._row(seq)
        prev = -1
        for p in range(L - 1, T):
            ctx = mx.array(taps[prev + 1 : p + 1].reshape(1, p - prev, 8 * H)).view(mx.bfloat16)
            anchor = mx.array([int(tokens[p + 1])], dtype=mx.int32)
            captured.clear()
            toks, _ = drafter._draft_batched([(st, row, ctx, anchor, None)])[0]
            mx.eval(toks, captured)
            props[p] = [int(t.item()) for t in toks]
            if len(saved) < a.save_logits:
                saved.append(np.array(mx.stack([c[0] for c in captured]).astype(mx.bfloat16).view(mx.uint16)))
            prev = p
        np.save(a.out / f"seq-{seq:05d}.props.npy", props)
        if saved:
            np.save(a.out / f"seq-{seq:05d}.logits.npy", np.stack(saved))
        lengths[seq] = accept_lengths(props, tokens, L - 1, T - 7)
        print(json.dumps({"seq": seq, "set": meta["set"], "rows": int(len(lengths[seq])),
                          "mean_len": float(lengths[seq].mean())}), flush=True)
    summary = {"quantized": a.quantized, "sets": {}}
    for kind in ("bench", "chat", "agentic", "all"):
        segs = [lengths[x["seq"]] for x in prompts if kind == "all" or x["set"] == kind]
        if not segs:
            continue
        summary["sets"][kind] = {"per_position": per_position(np.concatenate(segs)),
                                 "simulate": {d: {"tokens_per_cycle": float(np.mean([simulate(g, d)["tokens_per_cycle"] for g in segs]))}
                                              for d in (3, 5, 7)}}
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["sets"].get("all")))


if __name__ == "__main__":
    main()
