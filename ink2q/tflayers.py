"""TensorFold's layer gate. Per-layer hidden states of one sequence through the ink2q forward with an EXL3 pack's
experts (the reference) and through a second correct implementation of it (its static short convolutions in fp32:
the floor); a TensorFold dump passes a layer when its distance to the reference is at most max_ratio times the
floor's."""
import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models import inkling2

from ink2q import experts as X
from ink2q import model as M
from ink2q import stream as S
from ink2q.memguard import Floor, apply_limits
from ink2q.source import SourceIndex
from ink2q.tfio import bf16_bits_to_f32
from ink2q.tokens import load_manifest, load_tokens

MAX_RATIO = 1.5


@contextlib.contextmanager
def fp32_convs():
    """ShortConvolution with its window and weights in fp32 (mlx-lm convolves in the weights' bf16)."""
    original = inkling2.ShortConvolution.__call__

    def call(self, hidden_states, conv_state, conv_mask):
        k = self.kernel_size
        residual = hidden_states.astype(mx.float32)
        window = inkling2._conv_input(hidden_states, conv_state, conv_mask, k)
        out = mx.conv1d(window, self.conv1d.weight.astype(mx.float32), groups=self.conv1d.groups)
        return ((out + residual) / math.sqrt(k + 1)).astype(hidden_states.dtype), window

    inkling2.ShortConvolution.__call__ = call
    try:
        yield
    finally:
        inkling2.ShortConvolution.__call__ = original


def reference(src: SourceIndex, experts_fn, ids: np.ndarray, layers: int) -> tuple[np.ndarray, np.ndarray]:
    """experts_fn(layer) -> ExpertSpec | None replaces each layer's experts; both passes share each loaded layer."""
    m = M.build(src.config())
    M.load_globals(m, src, final=False)
    ref = S.embed(m, mx.array(ids))
    flo = S.SeqState(ref.h, None)
    out_r, out_f = [], []
    for layer in range(layers):
        M.load_layer(m, src, layer)
        mod = m.model.language_model.layers[layer]
        mx.eval(mod.parameters())
        spec = experts_fn(layer)
        with X.install(mod, spec) if spec is not None else contextlib.nullcontext():
            ref = S.step(m, layer, ref)
            mx.eval(ref.h, ref.blocks)
            with fp32_convs():
                flo = S.step(m, layer, flo)
                mx.eval(flo.h, flo.blocks)
        del spec
        M.drop_layer(m, layer)
        mx.clear_cache()
        out_r.append(np.array(ref.h[0].astype(mx.float32)))
        out_f.append(np.array(flo.h[0].astype(mx.float32)))
        print(json.dumps({"layer": layer}), flush=True)
    return np.stack(out_r), np.stack(out_f)


def rel_rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).mean() / (b**2).mean()))


def gate(ref: np.ndarray, floor: np.ndarray, tf: np.ndarray, max_ratio: float = MAX_RATIO) -> dict:
    rows = []
    for layer in range(tf.shape[0]):
        fl, d = rel_rms(floor[layer], ref[layer]), rel_rms(tf[layer], ref[layer])
        rows.append({"layer": layer, "rel_rms": d, "noise_floor": fl, "ratio": d / fl if fl > 0 else math.inf,
                     "max_abs": float(np.abs(tf[layer] - ref[layer]).max())})
    return {"layers": rows, "max_ratio": max_ratio, "pass": all(r["ratio"] <= max_ratio for r in rows)}


def read_dump(path: Path, tokens: int, hidden: int) -> np.ndarray:
    return bf16_bits_to_f32(np.fromfile(path, "<u2")).reshape(-1, tokens, hidden)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("ref")
    r.add_argument("--source", type=Path, required=True)
    r.add_argument("--kortexa", type=Path, required=True)
    r.add_argument("--tokens", type=Path, required=True)
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--case", default="en_web")
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--layers", type=int, required=True)
    r.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("gate")
    g.add_argument("--ref", type=Path, required=True)
    g.add_argument("--tf", type=Path, required=True)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--max-ratio", type=float, default=MAX_RATIO)
    a = ap.parse_args(argv)
    if a.cmd == "ref":
        from ink2q.exl3 import kortexa_spec
        apply_limits(150, 4)
        floor = Floor(24)
        seqs, man = load_tokens(a.tokens), load_manifest(a.manifest)
        i = next(k for k, m in enumerate(man) if m.get("case") == a.case)
        ids = seqs[i][: a.max_tokens]

        def experts_fn(layer):
            floor.check(f"layer {layer}")
            return kortexa_spec(a.kortexa, layer)

        ref, flo = reference(SourceIndex(a.source), experts_fn, ids, a.layers)
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "ids.json").write_text(json.dumps([int(t) for t in ids]))
        np.save(a.out / "ref.npy", ref)
        np.save(a.out / "floor.npy", flo)
        print(json.dumps({"case": a.case, "tokens": len(ids), "layers": a.layers}))
        return
    ref, flo = np.load(a.ref / "ref.npy"), np.load(a.ref / "floor.npy")
    tf = read_dump(a.tf, ref.shape[1], ref.shape[2])
    res = gate(ref[: tf.shape[0]], flo[: tf.shape[0]], tf, a.max_ratio)
    for row in res["layers"]:
        print(json.dumps(row))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({"layer_gate_pass": res["pass"]}))
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
